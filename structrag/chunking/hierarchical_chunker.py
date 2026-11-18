"""
structrag/chunking/hierarchical_chunker.py
-------------------------------------------
Hierarchical chunking pipeline — the main architectural contribution.

Core idea
---------
One chunk = one complete clause (or subclause) + all of its direct child
content: numbered paragraphs, notes, and table/figure placeholders.
The heading and full ancestor path are prepended to the chunk text so that
the embedding captures both the structural location and the content.

Example chunk text for subclause 6.2.2:

    [Eurocode 2 > Section 6 > 6.2 Effort tranchant > 6.2.2]
    6.2.2  Éléments ne nécessitant pas d'armatures d'effort tranchant

    (1)P Pour les membres ne nécessitant pas d'armatures de cisaillement ...
    (2)P La résistance au cisaillement de calcul VRd,c est donnée par ...
    NOTE  La valeur de CRd,c à utiliser en France est ...

Why this outperforms flat chunking on technical standards
----------------------------------------------------------
1. The ancestor path travels with every chunk — the embedding knows this
   discusses shear resistance in Section 6 of Eurocode 2.
2. Applicability conditions in (1)P and (2)P stay with the formula they
   govern — a flat window would split them across a page boundary.
3. Notes containing national annex values remain attached to their clause.
4. Cross-references extracted by the parser are preserved per chunk.

Token budget
------------
If a subclause's text exceeds MAX_TOKENS after assembly, it is split at
paragraph boundaries (never mid-sentence).  The heading block is repeated
at the start of each continuation chunk so every chunk is self-contained.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.config import HIER_MAX_TOKENS, HIER_OVERLAP_TOKENS
from structrag.parsing.hierarchy_schema import Node, load_nodes

MAX_TOKENS     = HIER_MAX_TOKENS
OVERLAP_TOKENS = HIER_OVERLAP_TOKENS
CHUNK_LEVELS   = {0, 1, 2, 3}  # section, clause, subclause, sub-subclause


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
# Index helpers
# ---------------------------------------------------------------------------

def _build_children_index(nodes: list[Node]) -> dict[str, list[Node]]:
    """Return {parent_id: [child_nodes]} for fast lookup."""
    index: dict[str, list[Node]] = {}
    for node in nodes:
        pid = node.parent_id or "__root__"
        index.setdefault(pid, []).append(node)
    return index


# ---------------------------------------------------------------------------
# Text assembly helpers
# ---------------------------------------------------------------------------

def _heading_block(node: Node) -> str:
    """Context prefix prepended to every chunk."""
    num   = node.clause_number or ""
    title = node.title or ""
    heading = f"{num}  {title}".strip() if (num or title) else ""
    return f"[{node.ancestor_path}]\n{heading}".strip()


def _child_text(child: Node) -> str:
    """Convert a child node (paragraph / note / table) to a text snippet."""
    tag = child.metadata.get("paragraph_tag", "") if child.metadata else ""
    if child.type == "paragraph":
        return f"{tag}  {child.text}".strip() if tag else child.text.strip()
    if child.type == "note":
        return f"NOTE  {child.text}".strip() if child.text else ""
    if child.type in ("table", "image"):
        return f"[TABLE/FIGURE — page {child.page_number}]"
    return child.text.strip() if child.text else ""


# ---------------------------------------------------------------------------
# Split oversized chunks at paragraph boundaries
# ---------------------------------------------------------------------------

def _split_into_parts(
    heading_block: str,
    child_texts: list[str],
    node: Node,
    source_id: str,
) -> list[dict]:
    """
    If the assembled text fits within MAX_TOKENS, return one chunk.
    Otherwise, split at paragraph boundaries and return multiple chunks,
    each prefixed with the heading block for context continuity.
    """
    parts: list[dict] = []
    part_index = 0
    current_texts: list[str] = []
    current_tokens = count_tokens(heading_block)

    def _make_chunk(texts: list[str], idx: int) -> dict:
        body = "\n\n".join(t for t in texts if t)
        full_text = f"{heading_block}\n\n{body}" if body else heading_block
        all_refs: list[str] = []
        for t in texts:
            all_refs.extend(re.findall(r'\b(\d+(?:\.\d+){1,3})\b', t))
        return {
            "chunk_id":        f"{node.id}_part{idx}" if idx > 0 else node.id,
            "source_id":       source_id,
            "chunk_type":      "hierarchical",
            "heading_node_id": node.id,
            "clause_number":   node.clause_number,
            "level":           node.level,
            "ancestor_path":   node.ancestor_path,
            "page_number":     node.page_number,
            "text":            full_text,
            "token_count":     count_tokens(full_text),
            "cross_refs":      list(dict.fromkeys(all_refs + node.cross_refs)),
            "part_index":      idx,
        }

    for ct in child_texts:
        ct_tokens = count_tokens(ct)
        if current_tokens + ct_tokens > MAX_TOKENS and current_texts:
            parts.append(_make_chunk(current_texts, part_index))
            part_index += 1
            current_texts = [ct]
            current_tokens = count_tokens(heading_block) + ct_tokens
        else:
            current_texts.append(ct)
            current_tokens += ct_tokens

    if current_texts or part_index == 0:
        parts.append(_make_chunk(current_texts, part_index))

    return parts


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

def chunk_hierarchical(nodes: list[Node], source_id: str) -> list[dict]:
    """
    Build hierarchical chunks from a parsed node list.

    Strategy
    --------
    1. Index nodes by parent_id.
    2. Walk all heading nodes (section, clause, subclause).
    3. Collect each heading's direct text children (paragraphs, notes, tables).
    4. Assemble heading_block + child texts into one chunk (or multiple if
       the token budget is exceeded).
    5. Skip headings whose only content is other headings — they are covered
       by their children's chunks.

    Parameters
    ----------
    nodes     : flat list of Node objects from standards_parser.py
    source_id : document identifier stored in each chunk (e.g. "ec2_2004_nf")

    Returns
    -------
    List of chunk dicts ready for indexer.py
    """
    children_idx = _build_children_index(nodes)
    chunks: list[dict] = []
    heading_types = {"section", "clause", "subclause"}

    for node in nodes:
        if node.type not in heading_types:
            continue

        direct_children = children_idx.get(node.id, [])
        text_children = [
            c for c in direct_children
            if c.type in ("paragraph", "note", "table", "image")
        ]

        if not text_children:
            continue  # pure structural container — content lives in its subclauses

        heading_blk = _heading_block(node)
        child_texts = [_child_text(c) for c in text_children]
        parts = _split_into_parts(heading_blk, child_texts, node, source_id)
        chunks.extend(parts)

    return chunks


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} hierarchical chunks -> {path}")


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import Counter

    from structrag.config import HIER_CHUNKS_PATH, NODES_PATH

    if not NODES_PATH.exists():
        print(f"ERROR: {NODES_PATH} not found. Run run_pipeline.py --from 1 first.")
        sys.exit(1)

    print(f"Loading nodes from {NODES_PATH.name} ...")
    nodes = load_nodes(str(NODES_PATH))
    print(f"Loaded {len(nodes)} nodes")

    print("Building hierarchical chunks ...")
    chunks = chunk_hierarchical(nodes, source_id="ec2_2004_nf")

    token_counts = [c["token_count"] for c in chunks]
    level_counts = Counter(c["level"] for c in chunks)
    part_counts  = Counter(c["part_index"] for c in chunks)

    print(f"\nTotal hierarchical chunks : {len(chunks)}")
    print(f"  Token stats — min:{min(token_counts)}  "
          f"avg:{sum(token_counts)//len(token_counts)}  "
          f"max:{max(token_counts)}")
    print(f"  By heading level : {dict(sorted(level_counts.items()))}")
    print(f"  Multi-part splits: {sum(v for k, v in part_counts.items() if k > 0)}")

    for c in chunks[:3]:
        print(f"\n  id={c['chunk_id']}")
        print(f"  path={c['ancestor_path'][:70]}")
        print(f"  tokens={c['token_count']}  part={c['part_index']}")
        print(f"  text preview: {c['text'][:200]}")

    save_chunks(chunks, str(HIER_CHUNKS_PATH))
