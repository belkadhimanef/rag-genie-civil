"""
structrag/retrieval/indexer.py
-------------------------------
Ingestion side of the retrieval layer.

Reads chunk JSON files produced by the chunkers and upserts them into the
appropriate ChromaDB collections with full metadata.

ChromaDB metadata fields stored per chunk
------------------------------------------
Hierarchical / child chunks:
    source_id, chunk_type, clause_number, level, ancestor_path,
    page_number, token_count, part_index, cross_refs (comma-separated str)

Flat chunks:
    source_id, chunk_type, start_page, end_page, token_count, part_index

Design notes
------------
- ChromaDB metadata values must be str, int, float, or bool — no lists.
  cross_refs is serialised as a comma-separated string and parsed back on
  retrieval.
- ancestor_path is stored both in metadata AND prepended to the chunk text,
  so the embedding captures it and metadata filters can target it.
- Batch size 100 keeps memory usage predictable on large corpora.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.config import (
    CHILD_CHUNKS_PATH,
    FLAT_CHUNKS_PATH,
    HIER_CHUNKS_PATH,
)
from structrag.retrieval.chroma_client import (
    COLLECTION_CHILD,
    COLLECTION_FLAT,
    COLLECTION_HIERARCHICAL,
    get_collection,
)

BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Metadata serialisers
# ---------------------------------------------------------------------------

def _meta_hierarchical(chunk: dict) -> dict[str, Any]:
    """Flatten hierarchical chunk fields to ChromaDB-compatible scalar types."""
    return {
        "source_id":     chunk.get("source_id", ""),
        "chunk_type":    "hierarchical",
        "clause_number": chunk.get("clause_number") or "",
        "level":         int(chunk.get("level", -1)),
        "ancestor_path": chunk.get("ancestor_path", ""),
        "page_number":   int(chunk.get("page_number", 0)),
        "token_count":   int(chunk.get("token_count", 0)),
        "part_index":    int(chunk.get("part_index", 0)),
        "cross_refs":    ",".join(chunk.get("cross_refs") or []),
    }


def _meta_flat(chunk: dict) -> dict[str, Any]:
    """Flatten flat chunk fields to ChromaDB-compatible scalar types."""
    return {
        "source_id":   chunk.get("source_id", ""),
        "chunk_type":  "flat",
        "start_page":  int(chunk.get("start_page", 0)),
        "end_page":    int(chunk.get("end_page", 0)),
        "token_count": int(chunk.get("token_count", 0)),
        "part_index":  int(chunk.get("part_index", 0)),
    }


def _meta_child(chunk: dict) -> dict[str, Any]:
    """Flatten child chunk fields to ChromaDB-compatible scalar types."""
    return {
        "source_id":       chunk.get("source_id", ""),
        "chunk_type":      "child",
        "parent_chunk_id": chunk.get("parent_chunk_id", ""),
        "clause_number":   chunk.get("clause_number", ""),
        "level":           int(chunk.get("level", -1)),
        "ancestor_path":   chunk.get("ancestor_path", ""),
        "page_number":     int(chunk.get("page_number", 0)),
        "token_count":     int(chunk.get("token_count", 0)),
        "child_type":      chunk.get("child_type", ""),
        "paragraph_tag":   chunk.get("paragraph_tag", ""),
    }


# ---------------------------------------------------------------------------
# Generic batch upsert
# ---------------------------------------------------------------------------

def _index_chunks(
    chunks: list[dict],
    collection_name: str,
    meta_fn,
    label: str,
) -> None:
    """Upsert all chunks into the named ChromaDB collection in batches."""
    collection = get_collection(collection_name)
    total = len(chunks)
    print(f"Indexing {total} {label} chunks into '{collection_name}' ...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[meta_fn(c) for c in batch],
        )
        done = min(batch_start + BATCH_SIZE, total)
        print(f"  {done}/{total} upserted ...", end="\r")

    print(f"\nDone. '{collection_name}' now has {collection.count()} documents.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_hierarchical(chunks_path: str | None = None) -> None:
    """Load hierarchical chunks JSON and upsert into ChromaDB."""
    path = chunks_path or str(HIER_CHUNKS_PATH)
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    _index_chunks(chunks, COLLECTION_HIERARCHICAL, _meta_hierarchical, "hierarchical")


def index_flat(chunks_path: str | None = None) -> None:
    """Load flat chunks JSON and upsert into ChromaDB."""
    path = chunks_path or str(FLAT_CHUNKS_PATH)
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    _index_chunks(chunks, COLLECTION_FLAT, _meta_flat, "flat")


def index_child_chunks(chunks_path: str | None = None) -> None:
    """Load child chunks JSON and upsert into ChromaDB."""
    path = chunks_path or str(CHILD_CHUNKS_PATH)
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    _index_chunks(chunks, COLLECTION_CHILD, _meta_child, "child")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from structrag.config import FLAT_CHUNKS_PATH as _FLAT, HIER_CHUNKS_PATH as _HIER

    if not _HIER.exists():
        print(f"ERROR: {_HIER} not found. Run run_pipeline.py --from 2 first.")
        sys.exit(1)
    if not _FLAT.exists():
        print(f"ERROR: {_FLAT} not found. Run run_pipeline.py --from 3 first.")
        sys.exit(1)

    print("=== Indexing hierarchical chunks ===")
    index_hierarchical()

    print("\n=== Indexing flat chunks ===")
    index_flat()

    print("\n=== Collections in ChromaDB ===")
    from structrag.retrieval.chroma_client import list_collections
    for name in list_collections():
        print(f"  {name}")
