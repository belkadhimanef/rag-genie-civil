"""
structrag/retrieval/retriever.py
---------------------------------
Query side of the retrieval layer.

Provides three retrieval functions that target different ChromaDB collections:
  - retrieve_hierarchical() : ec2_hierarchical collection
  - retrieve_flat()         : ec2_flat collection (baseline)
  - retrieve_child()        : ec2_child collection (small-to-big pattern)

All three accept the same base arguments so the evaluation pipeline can call
them interchangeably.

Result format
-------------
Each result is a dict with at minimum:
    chunk_id  str    — unique chunk identifier
    text      str    — chunk text passed to the LLM prompt
    score     float  — cosine distance (lower = more similar)
    metadata  dict   — raw ChromaDB metadata

Hierarchical / child results additionally expose:
    ancestor_path, clause_number, level, page_number, cross_refs

Flat results additionally expose:
    start_page, end_page
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.config import DEFAULT_N_RESULTS
from structrag.retrieval.chroma_client import (
    COLLECTION_CHILD,
    COLLECTION_FLAT,
    COLLECTION_HIERARCHICAL,
    get_collection,
)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _query(
    collection_name: str,
    query_text: str,
    n_results: int,
    where: Optional[dict],
) -> list[dict]:
    """
    Execute a ChromaDB similarity query and return a normalised result list.
    """
    collection = get_collection(collection_name, create_if_missing=False)
    count = collection.count()
    if count == 0:
        return []

    kwargs: dict = {
        "query_texts": [query_text],
        "n_results":   min(n_results, count),
        "include":     ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    raw = collection.query(**kwargs)
    docs      = raw["documents"][0]
    metas     = raw["metadatas"][0]
    distances = raw["distances"][0]
    ids       = raw["ids"][0]

    results: list[dict] = []
    for doc, meta, dist, chunk_id in zip(docs, metas, distances, ids):
        entry: dict = {
            "chunk_id": chunk_id,
            "text":     doc,
            "score":    round(float(dist), 4),
            "metadata": meta,
        }

        chunk_type = meta.get("chunk_type", "")

        if chunk_type in ("hierarchical", "parent"):
            entry["ancestor_path"] = meta.get("ancestor_path", "")
            entry["clause_number"] = meta.get("clause_number", "")
            entry["level"]         = int(meta.get("level", -1))
            entry["page_number"]   = int(meta.get("page_number", 0))
            raw_refs = meta.get("cross_refs", "")
            entry["cross_refs"] = [r for r in raw_refs.split(",") if r]

        elif chunk_type == "child":
            entry["parent_chunk_id"] = meta.get("parent_chunk_id", "")
            entry["child_type"]      = meta.get("child_type", "")
            entry["paragraph_tag"]   = meta.get("paragraph_tag", "")
            entry["clause_number"]   = meta.get("clause_number", "")
            entry["ancestor_path"]   = meta.get("ancestor_path", "")
            entry["level"]           = int(meta.get("level", -1))
            entry["page_number"]     = int(meta.get("page_number", 0))

        elif chunk_type == "flat":
            entry["start_page"] = int(meta.get("start_page", 0))
            entry["end_page"]   = int(meta.get("end_page", 0))

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_hierarchical(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    source_id: Optional[str] = None,
    level: Optional[int] = None,
    clause_number: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve from the hierarchical collection with optional metadata filters.

    Parameters
    ----------
    query         : natural-language question or keyword string
    n_results     : number of results to return
    source_id     : restrict to one document (e.g. "ec2_2004_nf")
    level         : filter by heading level (0=section, 1=clause, 2=subclause)
    clause_number : exact clause lookup (e.g. "6.2.2")
    """
    filters = []
    if source_id:
        filters.append({"source_id": {"$eq": source_id}})
    if level is not None:
        filters.append({"level": {"$eq": level}})
    if clause_number:
        filters.append({"clause_number": {"$eq": clause_number}})

    where: Optional[dict] = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    return _query(COLLECTION_HIERARCHICAL, query, n_results, where)


def retrieve_child(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    source_id: Optional[str] = None,
    clause_number: Optional[str] = None,
    child_type: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve from the child collection (small, precise chunks).

    Use with resolve_parents() from parent_child_chunker.py to implement
    the small-to-big retrieval pattern:
      1. Retrieve precise child chunks by similarity.
      2. Look up each child's parent chunk for the full clause context.
      3. Pass parent texts to the LLM.
    """
    filters = []
    if source_id:
        filters.append({"source_id": {"$eq": source_id}})
    if clause_number:
        filters.append({"clause_number": {"$eq": clause_number}})
    if child_type:
        filters.append({"child_type": {"$eq": child_type}})

    where: Optional[dict] = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    return _query(COLLECTION_CHILD, query, n_results, where)


def retrieve_flat(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    source_id: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve from the flat baseline collection.

    Parameters
    ----------
    query     : natural-language question
    n_results : number of results
    source_id : filter by source document
    """
    where: Optional[dict] = None
    if source_id:
        where = {"source_id": {"$eq": source_id}}

    return _query(COLLECTION_FLAT, query, n_results, where)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_query = "What is the minimum concrete cover for a beam in exposure class XC2?"
    print(f"Query: {test_query}\n")

    print("=== Hierarchical results ===")
    for i, r in enumerate(
        retrieve_hierarchical(test_query, n_results=3, source_id="ec2_2004_nf"), 1
    ):
        print(f"  [{i}] score={r['score']}  clause={r.get('clause_number', '')}  "
              f"p={r.get('page_number', '')}")
        print(f"       {r.get('ancestor_path', '')[:70]}")
        print(f"       {r['text'][:200]}")
        print()

    print("=== Flat results ===")
    for i, r in enumerate(
        retrieve_flat(test_query, n_results=3, source_id="ec2_2004_nf"), 1
    ):
        print(f"  [{i}] score={r['score']}  pages={r.get('start_page', '')}-{r.get('end_page', '')}")
        print(f"       {r['text'][:200]}")
        print()
