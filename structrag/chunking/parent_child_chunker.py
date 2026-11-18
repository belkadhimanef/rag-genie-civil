"""
structrag/chunking/parent_child_chunker.py
-------------------------------------------
Parent-child (small-to-big) chunking strategy.

Architecture
------------
Standard RAG has a precision-recall tension:
  - Small chunks  → precise embedding match, but missing surrounding context
  - Large chunks  → rich generation context, but diluted embedding signal

The parent-child pattern resolves this by keeping two representations:

  CHILD chunks  — small, focused, embeddable (one paragraph or one table row)
                  These are indexed in ChromaDB for retrieval.

  PARENT chunks — the full clause containing those children, with heading and
                  all sibling paragraphs assembled.
                  These are NOT indexed for retrieval.  They are looked up by
                  parent_chunk_id after retrieval to form the generation context.

Query flow
----------
1. Embed query → retrieve top-k CHILD chunks by cosine similarity + BM25.
2. For each child, look up its parent chunk by parent_chunk_id.
3. De-duplicate parents (multiple children may share one parent).
4. Pass PARENT texts to the LLM — full clause context, not fragments.

Table row splitting
-------------------
Markdown tables from table_extractor.py are split into individual row-level
child chunks, each carrying the caption + header rows so the row is
self-contained.  The full table is the parent.

Output
------
Two JSON files in data/chunks/:
  ec2_child_chunks.json   — small chunks for embedding / retrieval
  ec2_parent_chunks.json  — full-clause chunks for LLM generation context
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.config import (
    CHILD_CHUNKS_PATH,
    CHILD_MAX_TOKENS,
    PARENT_CHUNKS_PATH,
    PARENT_MAX_TOKENS,
)
from structrag.parsing.hierarchy_schema import Node, load_nodes

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))

except ImportError:
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) / 0.75)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_children_index(nodes: list[Node]) -> dict[str, list[Node]]:
    idx: dict[str, list[Node]] = {}
    for node in nodes:
        pid = node.parent_id or "__root__"
        idx.setdefault(pid, []).append(node)
    return idx


def _heading_block(node: Node) -> str:
    num   = node.clause_number or ""
    title = node.title or ""
    heading = f"{num}  {title}".strip() if (num or title) else ""
    return f"[{node.ancestor_path}]\n{heading}".strip()


def _child_text(child: Node) -> str:
    tag = child.metadata.get("paragraph_tag", "") if child.metadata else ""
    if child.type == "paragraph":
        return f"{tag}  {child.text}".strip() if tag else child.text.strip()
    if child.type == "note":
        return f"NOTE  {child.text}".strip() if child.text else ""
    if child.type in ("table", "image"):
        return f"[TABLE/FIGURE — page {child.page_number}]"
    return child.text.strip() if child.text else ""


def _extract_cross_refs(text: str) -> list[str]:
    refs = re.findall(r'\b(\d+(?:\.\d+){1,3})\b', text)
    return list(dict.fromkeys(refs))


def _split_parent_text(
    heading_block: str,
    child_texts: list[str],
    max_tokens: int = PARENT_MAX_TOKENS,
) -> list[str]:
    """Split a long clause into multiple parent texts at paragraph boundaries."""
    parts: list[str] = []
    current: list[str] = []
    current_tokens = count_tokens(heading_block)

    for ct in child_texts:
        ct_tokens = count_tokens(ct)
        if current_tokens + ct_tokens > max_tokens and current:
            parts.append(heading_block + "\n\n" + "\n\n".join(current))
            current = [ct]
            current_tokens = count_tokens(heading_block) + ct_tokens
        else:
            current.append(ct)
            current_tokens += ct_tokens

    if current or not parts:
        parts.append(heading_block + "\n\n" + "\n\n".join(current))

    return parts


def _split_table_rows(table_text: str, ancestor_path: str) -> list[str]:
    """
    Split a markdown table into individual row strings, each carrying the
    caption and header row so the row is self-contained for embedding.
    """
    lines = table_text.strip().split("\n")
    if not lines:
        return [table_text]

    caption_lines: list[str] = []
    header_row: Optional[str] = None
    sep_row:    Optional[str] = None
    data_rows:  list[str] = []

    i = 0
    while i < len(lines) and not lines[i].strip().startswith("|"):
        caption_lines.append(lines[i])
        i += 1
    if i < len(lines):
        header_row = lines[i]; i += 1
    if i < len(lines) and not set(lines[i].replace("|", "").replace("-", "").strip()):
        sep_row = lines[i]; i += 1
    while i < len(lines):
        if lines[i].strip():
            data_rows.append(lines[i])
        i += 1

    if not data_rows or header_row is None:
        return [table_text]

    caption_str = "\n".join(caption_lines)
    header_str  = f"{header_row}\n{sep_row}" if sep_row else header_row

    return [
        f"{caption_str}\n{header_str}\n{row}".strip()
        for row in data_rows
    ] or [table_text]


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

def chunk_parent_child(
    nodes: list[Node],
    source_id: str,
) -> tuple[list[dict], list[dict]]:
    """
    Build parent and child chunk lists from a parsed node list.

    Returns
    -------
    (child_chunks, parent_chunks)
    """
    children_idx  = _build_children_index(nodes)
    heading_types = {"section", "clause", "subclause"}

    parent_chunks: list[dict] = []
    child_chunks:  list[dict] = []

    for node in nodes:
        if node.type not in heading_types:
            continue

        direct_children = children_idx.get(node.id, [])
        text_children   = [
            c for c in direct_children
            if c.type in ("paragraph", "note", "table", "image")
        ]

        if not text_children:
            continue

        heading_blk = _heading_block(node)
        child_texts = [_child_text(c) for c in text_children]

        # ── Build PARENT chunks ──────────────────────────────────────────
        parent_parts = _split_parent_text(heading_blk, child_texts)
        parent_ids: list[str] = []

        for part_idx, part_text in enumerate(parent_parts):
            pid = f"{node.id}_parent" if part_idx == 0 else f"{node.id}_parent_p{part_idx}"
            all_refs = _extract_cross_refs(part_text) + node.cross_refs
            parent_chunks.append({
                "chunk_id":      pid,
                "source_id":     source_id,
                "chunk_type":    "parent",
                "clause_number": node.clause_number or "",
                "level":         node.level,
                "ancestor_path": node.ancestor_path,
                "page_number":   node.page_number,
                "text":          part_text,
                "token_count":   count_tokens(part_text),
                "cross_refs":    list(dict.fromkeys(all_refs)),
                "child_ids":     [],
                "part_index":    part_idx,
            })
            parent_ids.append(pid)

        # Map each child to the correct parent part
        children_per_part = max(1, len(text_children) // len(parent_parts))

        for child_idx, (child_node, child_text) in enumerate(
            zip(text_children, child_texts)
        ):
            part_idx_for_child = min(
                child_idx // children_per_part,
                len(parent_ids) - 1,
            )
            parent_id = parent_ids[part_idx_for_child]

            for p in parent_chunks:
                if p["chunk_id"] == parent_id:
                    p["child_ids"].append(f"{child_node.id}_child")
                    break

            # ── Build CHILD chunks ────────────────────────────────────────
            child_type = child_node.type

            if child_type == "table" and "|\n|" in child_text:
                row_texts = _split_table_rows(child_text, node.ancestor_path)
                for row_idx, row_text in enumerate(row_texts):
                    cid = f"{child_node.id}_child_row{row_idx}"
                    full_text = f"[{node.ancestor_path}]\n{row_text}"
                    child_chunks.append({
                        "chunk_id":        cid,
                        "source_id":       source_id,
                        "chunk_type":      "child",
                        "parent_chunk_id": parent_id,
                        "clause_number":   node.clause_number or "",
                        "level":           node.level,
                        "ancestor_path":   node.ancestor_path,
                        "page_number":     child_node.page_number,
                        "text":            full_text,
                        "token_count":     count_tokens(full_text),
                        "child_type":      "table_row",
                        "paragraph_tag":   "",
                    })
            else:
                cid = f"{child_node.id}_child"
                tag = child_node.metadata.get("paragraph_tag", "") \
                      if child_node.metadata else ""
                full_text = f"[{node.ancestor_path}]\n{child_text}"
                child_chunks.append({
                    "chunk_id":        cid,
                    "source_id":       source_id,
                    "chunk_type":      "child",
                    "parent_chunk_id": parent_id,
                    "clause_number":   node.clause_number or "",
                    "level":           node.level,
                    "ancestor_path":   node.ancestor_path,
                    "page_number":     child_node.page_number,
                    "text":            full_text,
                    "token_count":     count_tokens(full_text),
                    "child_type":      child_type,
                    "paragraph_tag":   tag,
                })

    return child_chunks, parent_chunks


# ---------------------------------------------------------------------------
# Table node injection
# ---------------------------------------------------------------------------

def _add_table_nodes_to_parent_child(
    pdf_path: str,
    source_id: str,
    child_chunks: list[dict],
    parent_chunks: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Inject pdfplumber-extracted table nodes into parent-child chunk lists.
    Each table becomes one parent chunk and multiple row-level child chunks.
    """
    from structrag.parsing.table_extractor import extract_tables

    table_nodes = extract_tables(pdf_path, source_id)

    for tnode in table_nodes:
        table_text = tnode.text
        tid        = tnode.id
        clause     = tnode.clause_number or ""
        path       = tnode.ancestor_path

        parent_id = f"{tid}_parent"
        parent_chunks.append({
            "chunk_id":      parent_id,
            "source_id":     source_id,
            "chunk_type":    "parent",
            "clause_number": clause,
            "level":         tnode.level,
            "ancestor_path": path,
            "page_number":   tnode.page_number,
            "text":          table_text,
            "token_count":   count_tokens(table_text),
            "cross_refs":    [],
            "child_ids":     [],
            "part_index":    0,
            "metadata":      tnode.metadata,
        })

        row_texts = _split_table_rows(table_text, path)
        for row_idx, row_text in enumerate(row_texts):
            cid = f"{tid}_child_row{row_idx}"
            full_text = f"[{path}]\n{row_text}"
            child_chunks.append({
                "chunk_id":        cid,
                "source_id":       source_id,
                "chunk_type":      "child",
                "parent_chunk_id": parent_id,
                "clause_number":   clause,
                "level":           tnode.level,
                "ancestor_path":   path,
                "page_number":     tnode.page_number,
                "text":            full_text,
                "token_count":     count_tokens(full_text),
                "child_type":      "table_row",
                "paragraph_tag":   "",
            })
            parent_chunks[-1]["child_ids"].append(cid)

    return child_chunks, parent_chunks


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks -> {path}")


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Parent lookup (used at generation time)
# ---------------------------------------------------------------------------

_parent_cache: Optional[dict[str, dict]] = None
_parent_cache_path: Optional[str] = None


def get_parent_lookup(parents_path: str) -> dict[str, dict]:
    """Return a {chunk_id: parent_chunk} dict, cached in module memory."""
    global _parent_cache, _parent_cache_path
    if _parent_cache is None or _parent_cache_path != parents_path:
        parents = load_chunks(parents_path)
        _parent_cache = {p["chunk_id"]: p for p in parents}
        _parent_cache_path = parents_path
    return _parent_cache


def resolve_parents(
    child_results: list[dict],
    parents_path: str,
) -> list[dict]:
    """
    Given child chunk retrieval results, return the corresponding parent
    chunks (de-duplicated, ordered by best child score).

    Parameters
    ----------
    child_results : results from retrieve_child()
    parents_path  : path to ec2_parent_chunks.json

    Returns
    -------
    List of parent chunk dicts with an added 'score' field.
    """
    parent_lookup = get_parent_lookup(parents_path)
    seen: set[str] = set()
    resolved: list[dict] = []

    for child in child_results:
        pid = child.get("parent_chunk_id", "")
        if pid and pid not in seen:
            parent = parent_lookup.get(pid)
            if parent:
                parent_copy = dict(parent)
                parent_copy["score"] = child.get("score", 0)
                parent_copy["retrieved_via_child"] = child["chunk_id"]
                resolved.append(parent_copy)
                seen.add(pid)

    return resolved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from collections import Counter

    from structrag.config import NODES_PATH

    pdf_candidates = list(PROJECT_ROOT.glob("*.pdf"))
    if not pdf_candidates:
        print("ERROR: No PDF found in project root.")
        sys.exit(1)
    pdf_path = pdf_candidates[0]

    if not NODES_PATH.exists():
        print(f"ERROR: {NODES_PATH} not found. Run run_pipeline.py --from 1 first.")
        sys.exit(1)

    print("Loading nodes ...")
    nodes = load_nodes(str(NODES_PATH))
    print(f"Loaded {len(nodes)} nodes")

    print("Building parent-child chunks ...")
    child_chunks, parent_chunks = chunk_parent_child(nodes, source_id="ec2_2004_nf")

    print("Injecting table nodes ...")
    child_chunks, parent_chunks = _add_table_nodes_to_parent_child(
        str(pdf_path), "ec2_2004_nf", child_chunks, parent_chunks
    )

    child_types   = Counter(c["child_type"] for c in child_chunks)
    parent_tokens = [p["token_count"] for p in parent_chunks]
    child_tokens  = [c["token_count"]  for c in child_chunks]

    print(f"\nParent chunks : {len(parent_chunks)}")
    print(f"  tokens — min:{min(parent_tokens)}  "
          f"avg:{sum(parent_tokens)//len(parent_tokens)}  "
          f"max:{max(parent_tokens)}")
    print(f"\nChild  chunks : {len(child_chunks)}")
    print(f"  tokens — min:{min(child_tokens)}  "
          f"avg:{sum(child_tokens)//len(child_tokens)}  "
          f"max:{max(child_tokens)}")
    print(f"  by type: {dict(child_types)}")

    save_chunks(child_chunks,  str(CHILD_CHUNKS_PATH))
    save_chunks(parent_chunks, str(PARENT_CHUNKS_PATH))
    print("\nDone.")
