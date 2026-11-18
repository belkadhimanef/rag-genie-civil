"""
structrag/chunking/flat_chunker.py
------------------------------------
Flat sliding-window baseline chunker.

This is the BASELINE that the hierarchical pipeline is compared against.
It uses the same corpus, the same embedding model, and the same token
budget.  The only difference is: no hierarchy awareness, no ancestor_path,
and no clause boundaries are respected.

How it works
------------
1. Read raw PDF text page-by-page with PyMuPDF.
2. Strip page headers/footers; concatenate all content pages.
3. Tokenise the full concatenated text.
4. Slide a window of MAX_TOKENS with OVERLAP_TOKENS stride.
5. Each window becomes one chunk — it will routinely span clause boundaries.

This deliberate structure-blindness is what makes it a valid baseline:
any measured difference in answer quality vs the hierarchical pipeline is
attributable solely to the chunking strategy.

Output
------
Each chunk dict:
    chunk_id    str   — unique identifier
    source_id   str   — document identifier
    chunk_type  str   — "flat"
    text        str   — raw window text
    token_count int
    start_page  int   — 1-indexed page of first token in window
    end_page    int   — 1-indexed page of last token in window
    part_index  int   — sequential window index
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.config import (
    FLAT_CHUNKS_PATH,
    FLAT_MAX_TOKENS,
    FLAT_OVERLAP_TOKENS,
    FLAT_SKIP_PAGES,
)

MAX_TOKENS     = FLAT_MAX_TOKENS
OVERLAP_TOKENS = FLAT_OVERLAP_TOKENS
SKIP_PAGES     = FLAT_SKIP_PAGES

# Page header pattern for the EC2 NF edition
_HEADER_RE    = re.compile(r'^Page\s+\d+')
_STANDARD_REF = "EN 1992-1-1:2004"

# ---------------------------------------------------------------------------
# Token counting with tiktoken fallback
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))

    def _encode(text: str) -> list:
        return _enc.encode(text)

    def _decode(token_ids: list) -> str:
        return _enc.decode(token_ids)

    _TIKTOKEN = True

except ImportError:
    _TIKTOKEN = False

    def count_tokens(text: str) -> int:  # type: ignore[misc]
        return int(len(text) / 4)

    def _encode(text: str) -> list:  # type: ignore[misc]
        return text.split()

    def _decode(token_ids: list) -> str:  # type: ignore[misc]
        if token_ids and isinstance(token_ids[0], str):
            return " ".join(token_ids)
        raise NotImplementedError("tiktoken unavailable — cannot decode token IDs")


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def _extract_pages(pdf_path: str, skip_before: int = SKIP_PAGES) -> list[tuple[int, str]]:
    """
    Return [(page_number_1indexed, cleaned_text), ...] for all content pages.

    Strips the EC2 running header/footer ("Page N" and standard reference line)
    and collapses internal whitespace.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages = []
    for i in range(skip_before, len(doc)):
        raw = doc[i].get_text("text")
        lines = [
            ln for ln in raw.splitlines()
            if not _HEADER_RE.match(ln.strip())
            and ln.strip() != _STANDARD_REF
            and ln.strip()
        ]
        clean_text = re.sub(r'\s+', ' ', " ".join(lines)).strip()
        if clean_text:
            pages.append((i + 1, clean_text))   # 1-indexed page number
    return pages


# ---------------------------------------------------------------------------
# Sliding-window chunker
# ---------------------------------------------------------------------------

def _sliding_window_chunks(
    pages: list[tuple[int, str]],
    source_id: str,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[dict]:
    """
    Concatenate all pages into one token stream and slide a fixed window.

    A sparse page-boundary table maps token positions back to 1-indexed page
    numbers so each chunk records start_page / end_page even though the
    chunker itself is entirely unaware of page structure.
    """
    # Build concatenated text and record char offset of each page start
    combined_parts: list[str] = []
    page_char_offsets: list[tuple[int, int]] = []   # (char_start, page_num)
    cursor = 0
    for page_num, text in pages:
        page_char_offsets.append((cursor, page_num))
        combined_parts.append(text)
        cursor += len(text) + 1   # +1 for the space separator
    combined_text = " ".join(combined_parts)

    # Build sparse (token_index, page_num) map at page boundaries
    all_tokens = _encode(combined_text)
    page_token_starts: list[tuple[int, int]] = []

    if _TIKTOKEN:
        import tiktoken as _tiktoken
        _local_enc = _tiktoken.get_encoding("cl100k_base")
        for char_start, page_num in page_char_offsets:
            tok_idx = len(_local_enc.encode(combined_text[:char_start]))
            page_token_starts.append((tok_idx, page_num))
    else:
        for char_start, page_num in page_char_offsets:
            tok_idx = len(combined_text[:char_start].split())
            page_token_starts.append((tok_idx, page_num))

    def _page_for_token(tok_idx: int) -> int:
        page = page_token_starts[0][1]
        for start, pnum in page_token_starts:
            if tok_idx >= start:
                page = pnum
            else:
                break
        return page

    # Slide the window
    chunks: list[dict] = []
    step = max_tokens - overlap
    start = 0
    part_index = 0

    while start < len(all_tokens):
        end = min(start + max_tokens, len(all_tokens))
        window = all_tokens[start:end]
        text = _decode(window)
        start_page = _page_for_token(start)
        end_page   = _page_for_token(end - 1)

        chunks.append({
            "chunk_id":    f"{source_id}_flat_{part_index}",
            "source_id":   source_id,
            "chunk_type":  "flat",
            "text":        text,
            "token_count": len(window),
            "start_page":  start_page,
            "end_page":    end_page,
            "part_index":  part_index,
        })

        part_index += 1
        start += step
        if end == len(all_tokens):
            break

    return chunks


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} flat chunks -> {path}")


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Resolve PDF path: look in project root
    pdf_candidates = list(PROJECT_ROOT.glob("*.pdf"))
    if not pdf_candidates:
        print("ERROR: No PDF found in project root. Place the EC2 PDF there and retry.")
        sys.exit(1)
    pdf_path = pdf_candidates[0]
    print(f"Using PDF: {pdf_path.name}")

    print(f"Extracting text (skipping first {SKIP_PAGES} pages) ...")
    pages = _extract_pages(str(pdf_path), SKIP_PAGES)
    print(f"Content pages: {len(pages)}")

    print(f"Building flat chunks (window={MAX_TOKENS}, overlap={OVERLAP_TOKENS}) ...")
    chunks = _sliding_window_chunks(pages, source_id="ec2_2004_nf")

    token_counts = [c["token_count"] for c in chunks]
    print(f"\nTotal flat chunks : {len(chunks)}")
    print(f"Token stats — min:{min(token_counts)}  "
          f"avg:{sum(token_counts)//len(token_counts)}  "
          f"max:{max(token_counts)}")

    for c in chunks[:3]:
        print(f"\n  id={c['chunk_id']}  pages={c['start_page']}-{c['end_page']}  "
              f"tokens={c['token_count']}")
        print(f"  text: {c['text'][:200]}")

    save_chunks(chunks, str(FLAT_CHUNKS_PATH))
