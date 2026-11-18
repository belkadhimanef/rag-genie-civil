"""
structrag/retrieval/hybrid_retriever.py
-----------------------------------------
Hybrid BM25 + vector retrieval with Reciprocal Rank Fusion (RRF).

Motivation
----------
Embedding models encode semantics well but struggle with exact symbolic
queries such as "γc = 1.5" or "εcu2 3.5‰" — these are term-matching
problems, not semantic ones.  BM25 handles them reliably.

Combining both with RRF gives the best of both worlds:
  - BM25 catches exact symbols, clause numbers, and French technical terms.
  - Vector search catches semantically equivalent phrasings.
  - RRF merges the two ranked lists without requiring score normalisation.

RRF formula
-----------
    score_rrf(d) = 1/(k + rank_vector(d)) + 1/(k + rank_bm25(d))
    k = 60  (standard constant that dampens rank-position differences)

Query expansion
---------------
Queries containing exposure class codes (XC2, XD3, …) or structural class
codes (S1-S6) are automatically expanded for BM25 to improve recall on
Table 4.4N cover lookups.

BM25 index
----------
Built lazily on first call from the hierarchical chunks JSON.  French
stemming (NLTK Snowball) is applied when NLTK is available; falls back to
simple whitespace tokenisation otherwise.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.config import HIER_CHUNKS_PATH, RRF_K
from structrag.retrieval.retriever import retrieve_hierarchical

# ---------------------------------------------------------------------------
# BM25 index — built lazily, cached in module globals
# ---------------------------------------------------------------------------
_bm25_index = None
_bm25_chunks: list[dict] = []


def _tokenise(text: str) -> list[str]:
    """Tokenise text for BM25; apply French stemming when NLTK is available."""
    tokens = re.findall(r'[^\s\|\,\;\.\:\!\?\(\)\[\]\{\}]+', text.lower())
    try:
        from nltk.stem.snowball import FrenchStemmer
        stemmer = FrenchStemmer()
        return [stemmer.stem(t) for t in tokens]
    except ImportError:
        return tokens


def _build_bm25_index() -> None:
    """Load all hierarchical chunks and build an in-memory BM25 index."""
    global _bm25_index, _bm25_chunks

    if _bm25_index is not None:
        return

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("rank_bm25 not installed. Run: pip install rank-bm25")

    if not HIER_CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Hierarchical chunks not found: {HIER_CHUNKS_PATH}\n"
            "Run run_pipeline.py first to build the index."
        )

    with open(HIER_CHUNKS_PATH, encoding="utf-8") as f:
        _bm25_chunks = json.load(f)

    corpus = [_tokenise(c.get("text", "")) for c in _bm25_chunks]
    _bm25_index = BM25Okapi(corpus)
    stemming = "with French stemming" if _nltk_available() else "no stemming (install nltk)"
    print(f"[BM25] Index built: {len(_bm25_chunks)} documents ({stemming})")


def _nltk_available() -> bool:
    try:
        from nltk.stem.snowball import FrenchStemmer  # noqa: F401
        return True
    except ImportError:
        return False


def _bm25_search(query: str, n: int) -> list[dict]:
    """Return top-n BM25 results formatted like retriever output."""
    import heapq

    _build_bm25_index()
    tokens = _tokenise(query)
    scores = _bm25_index.get_scores(tokens)
    top_indices = heapq.nlargest(n, range(len(scores)), key=lambda i: scores[i])

    results = []
    for rank, idx in enumerate(top_indices):
        chunk = _bm25_chunks[idx]
        entry = {
            "chunk_id":     chunk.get("chunk_id", ""),
            "text":         chunk.get("text", ""),
            "score":        round(float(scores[idx]), 4),
            "bm25_rank":    rank + 1,
            "metadata":     {k: v for k, v in chunk.items() if k != "text"},
            "retrieval":    "bm25",
        }
        if chunk.get("chunk_type") == "hierarchical":
            entry["ancestor_path"] = chunk.get("ancestor_path", "")
            entry["clause_number"] = chunk.get("clause_number", "")
            entry["level"]         = int(chunk.get("level", -1))
            entry["page_number"]   = int(chunk.get("page_number", 0))
            raw_refs = chunk.get("cross_refs", [])
            entry["cross_refs"] = (
                raw_refs if isinstance(raw_refs, list)
                else [r for r in raw_refs.split(",") if r]
            )
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    vector_results: list[dict],
    bm25_results: list[dict],
    n_results: int,
) -> list[dict]:
    """Merge two ranked lists using RRF and return the top n_results items."""
    merged: dict[str, dict] = {}

    for rank, r in enumerate(vector_results, 1):
        cid = r["chunk_id"]
        if cid not in merged:
            merged[cid] = dict(r)
            merged[cid]["rrf_score"] = 0.0
            merged[cid]["found_by"] = []
        merged[cid]["rrf_score"] += 1.0 / (RRF_K + rank)
        merged[cid]["found_by"].append("vector")
        merged[cid]["vector_rank"] = rank

    for rank, r in enumerate(bm25_results, 1):
        cid = r["chunk_id"]
        if cid not in merged:
            merged[cid] = dict(r)
            merged[cid]["rrf_score"] = 0.0
            merged[cid]["found_by"] = []
        merged[cid]["rrf_score"] += 1.0 / (RRF_K + rank)
        merged[cid]["found_by"].append("bm25")
        merged[cid]["bm25_rank"] = rank

    ranked = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)

    for item in ranked:
        item["score"] = round(item["rrf_score"], 6)

    return ranked[:n_results]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_hybrid(
    query: str,
    n_results: int = 8,
    source_id: Optional[str] = None,
    level: Optional[int] = None,
    clause_number: Optional[str] = None,
) -> list[dict]:
    """
    Hybrid BM25 + vector retrieval with RRF re-ranking.

    Parameters
    ----------
    query         : user question or keyword string
    n_results     : number of final results to return
    source_id     : filter to one document (e.g. "ec2_2004_nf")
    level         : filter to one heading level
    clause_number : exact clause lookup
    """
    # Expand query for table-lookup queries containing class identifiers
    exposure_classes  = re.findall(r'\b(X[CDSFA][0-9])\b', query, re.IGNORECASE)
    structural_classes = re.findall(r'\b(S[1-6])\b', query, re.IGNORECASE)
    concrete_classes  = re.findall(r'\b(C\d{2}/\d{2})\b', query, re.IGNORECASE)

    if exposure_classes or structural_classes or concrete_classes:
        extras = " ".join(exposure_classes + structural_classes + concrete_classes)
        bm25_query = f"{query} {extras} enrobage cmin,dur Tableau 4.4N"
    else:
        bm25_query = query

    k_fetch = max(n_results * 2, 10)

    vector_results = retrieve_hierarchical(
        query=query,
        n_results=k_fetch,
        source_id=source_id,
        level=level,
        clause_number=clause_number,
    )

    try:
        bm25_results = _bm25_search(bm25_query, k_fetch)
    except (ImportError, FileNotFoundError) as exc:
        print(f"[BM25] Warning: {exc} — falling back to vector-only")
        return vector_results[:n_results]
    except Exception as exc:
        print(f"[BM25] Warning: {exc} — falling back to vector-only")
        return vector_results[:n_results]

    return _rrf_fuse(vector_results, bm25_results, n_results)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    test_queries = [
        "partial safety factor concrete gamma_c 1.5",
        "compressive strain epsilon_cu2 parabola rectangle",
        "minimum cover exposure class XC2 XC3",
        "structural classification Table 4.3N",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retrieve_hybrid(q, n_results=3, source_id="ec2_2004_nf")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] rrf={r['score']:.4f}  by={r.get('found_by', ['?'])}  "
                  f"clause={r.get('clause_number', '')}  "
                  f"path={r.get('ancestor_path', '')[:60]}")
