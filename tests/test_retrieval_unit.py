"""
tests/test_retrieval_unit.py
-----------------------------
Unit tests for retrieval helpers that do NOT require a live ChromaDB instance.

Tests cover:
- RRF fusion logic in hybrid_retriever.py
- Metadata serialisers in indexer.py
- Query builder (filter construction) in retriever.py
- Table row splitting utilities
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

class TestRRFFusion:
    def _make_result(self, chunk_id: str, score: float = 0.5) -> dict:
        return {
            "chunk_id":      chunk_id,
            "text":          f"text for {chunk_id}",
            "score":         score,
            "ancestor_path": "",
            "clause_number": "",
            "level":         2,
            "page_number":   1,
            "cross_refs":    [],
        }

    def test_fuse_two_lists_no_overlap(self):
        from structrag.retrieval.hybrid_retriever import _rrf_fuse
        vec   = [self._make_result(f"v{i}") for i in range(3)]
        bm25  = [self._make_result(f"b{i}") for i in range(3)]
        fused = _rrf_fuse(vec, bm25, n_results=5)
        assert len(fused) == 5
        ids = [r["chunk_id"] for r in fused]
        # Both lists should contribute
        assert any(i.startswith("v") for i in ids)
        assert any(i.startswith("b") for i in ids)

    def test_fuse_respects_n_results(self):
        from structrag.retrieval.hybrid_retriever import _rrf_fuse
        vec   = [self._make_result(f"v{i}") for i in range(10)]
        bm25  = [self._make_result(f"b{i}") for i in range(10)]
        fused = _rrf_fuse(vec, bm25, n_results=4)
        assert len(fused) == 4

    def test_shared_chunk_gets_higher_score(self):
        """A chunk appearing in both lists should rank above chunk-only-in-one-list."""
        from structrag.retrieval.hybrid_retriever import _rrf_fuse
        shared = self._make_result("shared")
        vec    = [shared] + [self._make_result(f"v{i}") for i in range(5)]
        bm25   = [shared] + [self._make_result(f"b{i}") for i in range(5)]
        fused  = _rrf_fuse(vec, bm25, n_results=10)
        # shared should be first
        assert fused[0]["chunk_id"] == "shared"

    def test_found_by_field_populated(self):
        from structrag.retrieval.hybrid_retriever import _rrf_fuse
        vec  = [self._make_result("v0")]
        bm25 = [self._make_result("b0")]
        fused = _rrf_fuse(vec, bm25, n_results=5)
        for r in fused:
            assert "found_by" in r
            assert isinstance(r["found_by"], list)

    def test_empty_bm25_returns_vector_results(self):
        from structrag.retrieval.hybrid_retriever import _rrf_fuse
        vec   = [self._make_result(f"v{i}") for i in range(3)]
        fused = _rrf_fuse(vec, [], n_results=5)
        assert len(fused) == 3

    def test_empty_vector_returns_bm25_results(self):
        from structrag.retrieval.hybrid_retriever import _rrf_fuse
        bm25  = [self._make_result(f"b{i}") for i in range(3)]
        fused = _rrf_fuse([], bm25, n_results=5)
        assert len(fused) == 3

    def test_scores_are_positive(self):
        from structrag.retrieval.hybrid_retriever import _rrf_fuse
        vec  = [self._make_result(f"v{i}") for i in range(3)]
        bm25 = [self._make_result(f"b{i}") for i in range(3)]
        for r in _rrf_fuse(vec, bm25, n_results=6):
            assert r["score"] > 0

    def test_results_sorted_descending(self):
        from structrag.retrieval.hybrid_retriever import _rrf_fuse
        vec  = [self._make_result(f"v{i}") for i in range(5)]
        bm25 = [self._make_result(f"b{i}") for i in range(5)]
        fused = _rrf_fuse(vec, bm25, n_results=8)
        scores = [r["score"] for r in fused]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Metadata serialisers
# ---------------------------------------------------------------------------

class TestMetaSerialiserHierarchical:
    def _chunk(self, **overrides) -> dict:
        base = {
            "chunk_id":      "ec2_2004_nf_subclause_6.2.2",
            "source_id":     "ec2_2004_nf",
            "chunk_type":    "hierarchical",
            "clause_number": "6.2.2",
            "level":         2,
            "ancestor_path": "Eurocode 2 > Section 6 > 6.2.2",
            "page_number":   87,
            "token_count":   312,
            "part_index":    0,
            "cross_refs":    ["4.4.1.2", "3.1.2"],
            "text":          "some text",
        }
        base.update(overrides)
        return base

    def test_all_values_are_scalar(self):
        from structrag.retrieval.indexer import _meta_hierarchical
        meta = _meta_hierarchical(self._chunk())
        for k, v in meta.items():
            assert isinstance(v, (str, int, float, bool)), (
                f"Key '{k}' has non-scalar type {type(v)}"
            )

    def test_cross_refs_serialised_as_string(self):
        from structrag.retrieval.indexer import _meta_hierarchical
        meta = _meta_hierarchical(self._chunk())
        assert isinstance(meta["cross_refs"], str)
        assert "4.4.1.2" in meta["cross_refs"]

    def test_level_is_int(self):
        from structrag.retrieval.indexer import _meta_hierarchical
        meta = _meta_hierarchical(self._chunk())
        assert isinstance(meta["level"], int)

    def test_missing_cross_refs_gives_empty_string(self):
        from structrag.retrieval.indexer import _meta_hierarchical
        chunk = self._chunk()
        del chunk["cross_refs"]
        meta = _meta_hierarchical(chunk)
        assert meta["cross_refs"] == ""


class TestMetaSerialiserFlat:
    def test_flat_meta_scalar_types(self):
        from structrag.retrieval.indexer import _meta_flat
        chunk = {
            "source_id":   "ec2_2004_nf",
            "chunk_type":  "flat",
            "start_page":  18,
            "end_page":    19,
            "token_count": 512,
            "part_index":  0,
            "text":        "some text",
        }
        meta = _meta_flat(chunk)
        for k, v in meta.items():
            assert isinstance(v, (str, int, float, bool))

    def test_flat_meta_has_page_range(self):
        from structrag.retrieval.indexer import _meta_flat
        chunk = {
            "source_id":   "ec2_2004_nf",
            "start_page":  18, "end_page": 20,
            "token_count": 300, "part_index": 0, "text": "x",
        }
        meta = _meta_flat(chunk)
        assert meta["start_page"] == 18
        assert meta["end_page"] == 20


class TestMetaSerialiserChild:
    def test_child_meta_has_parent_id(self):
        from structrag.retrieval.indexer import _meta_child
        chunk = {
            "source_id":       "ec2_2004_nf",
            "parent_chunk_id": "ec2_2004_nf_subclause_6.2.2_parent",
            "clause_number":   "6.2.2",
            "level":           2,
            "ancestor_path":   "Eurocode 2 > 6.2.2",
            "page_number":     87,
            "token_count":     45,
            "child_type":      "paragraph",
            "paragraph_tag":   "(1)P",
            "text":            "paragraph text",
        }
        meta = _meta_child(chunk)
        assert meta["parent_chunk_id"] == "ec2_2004_nf_subclause_6.2.2_parent"
        assert meta["child_type"] == "paragraph"
        assert isinstance(meta["level"], int)
