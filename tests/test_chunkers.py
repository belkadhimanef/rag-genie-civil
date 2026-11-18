"""
tests/test_chunkers.py
-----------------------
Unit tests for the three chunking strategies.

Tests use synthetic Node objects — no PDF or external dependencies required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from structrag.parsing.hierarchy_schema import Node


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _node(
    nid: str,
    ntype: str,
    level: int,
    clause: str | None,
    parent_id: str | None,
    text: str = "",
    title: str = "",
    ancestor_path: str = "Eurocode 2",
    page: int = 1,
    cross_refs: list[str] | None = None,
) -> Node:
    return Node(
        id=nid,
        source_id="ec2_test",
        type=ntype,
        level=level,
        title=title,
        clause_number=clause,
        text=text,
        parent_id=parent_id,
        ancestor_path=ancestor_path,
        page_number=page,
        cross_refs=cross_refs or [],
    )


def _build_simple_tree() -> list[Node]:
    """
    Build a minimal synthetic node tree:
        section_6  (level 0)
          clause_6.2  (level 1)
            subclause_6.2.2  (level 2)
              paragraph_1  (level 3)
              paragraph_2  (level 3)
              note_1       (level 3)
            subclause_6.2.3  (level 2)
              paragraph_3  (level 3)
    """
    sec = _node("sec_6", "section", 0, None,  None,       title="Section 6",
                ancestor_path="Eurocode 2 > Section 6")
    cl  = _node("cl_6_2", "clause", 1, "6.2", "sec_6",   title="Effort tranchant",
                ancestor_path="Eurocode 2 > Section 6 > 6.2")
    sc1 = _node("sc_6_2_2", "subclause", 2, "6.2.2", "cl_6_2",
                title="Sans armatures",
                ancestor_path="Eurocode 2 > Section 6 > 6.2 > 6.2.2")
    p1  = _node("p1", "paragraph", 3, None, "sc_6_2_2",
                text="(1)P La résistance de calcul VRd,c est: VRd,c = CRd,c × k × (100 ρl fck)^(1/3)")
    p2  = _node("p2", "paragraph", 3, None, "sc_6_2_2",
                text="(2)P Avec k = 1 + sqrt(200/d) ≤ 2,0 et ρl = Asl / (bw × d)")
    n1  = _node("n1", "note", 3, None, "sc_6_2_2",
                text="NOTE La valeur de CRd,c recommandée est 0,18/γc = 0,12")
    sc2 = _node("sc_6_2_3", "subclause", 2, "6.2.3", "cl_6_2",
                title="Avec armatures",
                ancestor_path="Eurocode 2 > Section 6 > 6.2 > 6.2.3")
    p3  = _node("p3", "paragraph", 3, None, "sc_6_2_3",
                text="(1)P La résistance de calcul VRd,s = (Asw/s) × z × fywd × cot θ")
    return [sec, cl, sc1, p1, p2, n1, sc2, p3]


# ---------------------------------------------------------------------------
# Hierarchical chunker tests
# ---------------------------------------------------------------------------

class TestHierarchicalChunker:
    def test_produces_chunks(self):
        from structrag.chunking.hierarchical_chunker import chunk_hierarchical
        nodes = _build_simple_tree()
        chunks = chunk_hierarchical(nodes, source_id="ec2_test")
        assert len(chunks) > 0

    def test_chunk_has_required_fields(self):
        from structrag.chunking.hierarchical_chunker import chunk_hierarchical
        nodes = _build_simple_tree()
        chunks = chunk_hierarchical(nodes, source_id="ec2_test")
        required = {"chunk_id", "source_id", "chunk_type", "text",
                    "token_count", "ancestor_path", "part_index"}
        for chunk in chunks:
            missing = required - set(chunk.keys())
            assert not missing, f"Chunk missing fields: {missing}"

    def test_chunk_type_is_hierarchical(self):
        from structrag.chunking.hierarchical_chunker import chunk_hierarchical
        nodes = _build_simple_tree()
        for chunk in chunk_hierarchical(nodes, source_id="ec2_test"):
            assert chunk["chunk_type"] == "hierarchical"

    def test_ancestor_path_in_text(self):
        """The ancestor path must be prepended to every chunk's text."""
        from structrag.chunking.hierarchical_chunker import chunk_hierarchical
        nodes = _build_simple_tree()
        for chunk in chunk_hierarchical(nodes, source_id="ec2_test"):
            assert chunk["ancestor_path"] in chunk["text"], (
                f"Ancestor path not found in chunk text for {chunk['chunk_id']}"
            )

    def test_no_empty_chunks(self):
        from structrag.chunking.hierarchical_chunker import chunk_hierarchical
        nodes = _build_simple_tree()
        for chunk in chunk_hierarchical(nodes, source_id="ec2_test"):
            assert chunk["text"].strip(), f"Empty chunk: {chunk['chunk_id']}"

    def test_pure_container_nodes_skipped(self):
        """Section node with no direct text children should not appear as a chunk."""
        from structrag.chunking.hierarchical_chunker import chunk_hierarchical
        nodes = _build_simple_tree()
        chunks = chunk_hierarchical(nodes, source_id="ec2_test")
        chunk_ids = [c["chunk_id"] for c in chunks]
        # sec_6 has no direct paragraph/note children — should be skipped
        assert "sec_6" not in chunk_ids

    def test_subclause_chunks_contain_child_text(self):
        """The paragraph text must appear inside the subclause chunk."""
        from structrag.chunking.hierarchical_chunker import chunk_hierarchical
        nodes = _build_simple_tree()
        chunks = chunk_hierarchical(nodes, source_id="ec2_test")
        sc_chunk = next((c for c in chunks if "sc_6_2_2" in c["chunk_id"]), None)
        assert sc_chunk is not None
        assert "VRd,c" in sc_chunk["text"]
        assert "NOTE" in sc_chunk["text"]

    def test_token_count_positive(self):
        from structrag.chunking.hierarchical_chunker import chunk_hierarchical
        nodes = _build_simple_tree()
        for chunk in chunk_hierarchical(nodes, source_id="ec2_test"):
            assert chunk["token_count"] > 0

    def test_save_load_round_trip(self, tmp_path):
        from structrag.chunking.hierarchical_chunker import (
            chunk_hierarchical, load_chunks, save_chunks,
        )
        nodes = _build_simple_tree()
        chunks = chunk_hierarchical(nodes, source_id="ec2_test")
        path = str(tmp_path / "hier.json")
        save_chunks(chunks, path)
        loaded = load_chunks(path)
        assert len(loaded) == len(chunks)
        assert loaded[0]["chunk_id"] == chunks[0]["chunk_id"]


# ---------------------------------------------------------------------------
# Parent-child chunker tests
# ---------------------------------------------------------------------------

class TestParentChildChunker:
    def test_produces_both_lists(self):
        from structrag.chunking.parent_child_chunker import chunk_parent_child
        nodes = _build_simple_tree()
        child_chunks, parent_chunks = chunk_parent_child(nodes, source_id="ec2_test")
        assert len(child_chunks) > 0
        assert len(parent_chunks) > 0

    def test_child_chunk_type(self):
        from structrag.chunking.parent_child_chunker import chunk_parent_child
        nodes = _build_simple_tree()
        child_chunks, _ = chunk_parent_child(nodes, source_id="ec2_test")
        for c in child_chunks:
            assert c["chunk_type"] == "child"

    def test_parent_chunk_type(self):
        from structrag.chunking.parent_child_chunker import chunk_parent_child
        nodes = _build_simple_tree()
        _, parent_chunks = chunk_parent_child(nodes, source_id="ec2_test")
        for p in parent_chunks:
            assert p["chunk_type"] == "parent"

    def test_every_child_has_parent_chunk_id(self):
        from structrag.chunking.parent_child_chunker import chunk_parent_child
        nodes = _build_simple_tree()
        child_chunks, parent_chunks = chunk_parent_child(nodes, source_id="ec2_test")
        parent_ids = {p["chunk_id"] for p in parent_chunks}
        for c in child_chunks:
            assert c["parent_chunk_id"] in parent_ids, (
                f"Child {c['chunk_id']} has unknown parent_chunk_id "
                f"'{c['parent_chunk_id']}'"
            )

    def test_resolve_parents(self, tmp_path):
        from structrag.chunking.parent_child_chunker import (
            chunk_parent_child, resolve_parents, save_chunks,
        )
        nodes = _build_simple_tree()
        child_chunks, parent_chunks = chunk_parent_child(nodes, source_id="ec2_test")
        parents_path = str(tmp_path / "parents.json")
        save_chunks(parent_chunks, parents_path)

        # Simulate retrieval: take first two child chunks
        result = resolve_parents(child_chunks[:2], parents_path)
        assert len(result) >= 1
        for r in result:
            assert r["chunk_type"] == "parent"
            assert "text" in r

    def test_child_tokens_smaller_than_parent_tokens(self):
        """Children should generally be smaller than parents."""
        from structrag.chunking.parent_child_chunker import chunk_parent_child
        nodes = _build_simple_tree()
        child_chunks, parent_chunks = chunk_parent_child(nodes, source_id="ec2_test")
        avg_child  = sum(c["token_count"] for c in child_chunks) / len(child_chunks)
        avg_parent = sum(p["token_count"] for p in parent_chunks) / len(parent_chunks)
        assert avg_child < avg_parent, (
            f"Expected avg child tokens ({avg_child:.0f}) < "
            f"avg parent tokens ({avg_parent:.0f})"
        )


# ---------------------------------------------------------------------------
# Table row splitting tests
# ---------------------------------------------------------------------------

class TestTableRowSplitting:
    _TABLE_MARKDOWN = (
        "Tableau 4.4N : Enrobage minimal cmin,dur (mm)\n\n"
        "| Classe structurale | XC1 | XC2 | XD1 |\n"
        "|---|---|---|---|\n"
        "| S1 | 10 | 10 | 20 |\n"
        "| S2 | 10 | 15 | 25 |\n"
        "| S3 | 10 | 20 | 30 |\n"
    )

    def test_splits_into_data_rows(self):
        from structrag.chunking.parent_child_chunker import _split_table_rows
        rows = _split_table_rows(self._TABLE_MARKDOWN, "Eurocode 2 > Table 4.4N")
        assert len(rows) == 3  # S1, S2, S3

    def test_each_row_contains_header(self):
        from structrag.chunking.parent_child_chunker import _split_table_rows
        rows = _split_table_rows(self._TABLE_MARKDOWN, "Eurocode 2 > Table 4.4N")
        for row in rows:
            assert "Classe structurale" in row

    def test_each_row_contains_caption(self):
        from structrag.chunking.parent_child_chunker import _split_table_rows
        rows = _split_table_rows(self._TABLE_MARKDOWN, "Eurocode 2 > Table 4.4N")
        for row in rows:
            assert "Tableau 4.4N" in row

    def test_fallback_on_non_table(self):
        from structrag.chunking.parent_child_chunker import _split_table_rows
        plain = "This is not a table."
        rows = _split_table_rows(plain, "some path")
        assert rows == [plain]
