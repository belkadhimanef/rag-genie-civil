"""
tests/test_hierarchy_schema.py
-------------------------------
Unit tests for the Node dataclass and serialisation helpers.

No PDF, API keys, or ChromaDB instance required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from structrag.parsing.hierarchy_schema import Node, load_nodes, save_nodes


def _make_node(**overrides) -> Node:
    """Return a minimal valid Node, optionally overriding fields."""
    defaults = dict(
        id="ec2_2004_nf_subclause_6.2.2",
        source_id="ec2_2004_nf",
        type="subclause",
        level=2,
        title="Éléments ne nécessitant pas d'armatures d'effort tranchant",
        clause_number="6.2.2",
        text="(1)P La résistance au cisaillement de calcul VRd,c est donnée par ...",
        parent_id="ec2_2004_nf_clause_6.2",
        ancestor_path="Eurocode 2 > Section 6 > 6.2 Effort tranchant > 6.2.2",
        page_number=87,
    )
    defaults.update(overrides)
    return Node(**defaults)


class TestNodeCreation:
    def test_basic_creation(self):
        node = _make_node()
        assert node.id == "ec2_2004_nf_subclause_6.2.2"
        assert node.clause_number == "6.2.2"
        assert node.level == 2

    def test_default_lists_are_empty(self):
        node = _make_node()
        assert node.cross_refs == []
        assert node.linked_node_ids == []
        assert node.metadata == {}

    def test_cross_refs_stored(self):
        node = _make_node(cross_refs=["6.2.3", "Table 4.4N"])
        assert "6.2.3" in node.cross_refs
        assert len(node.cross_refs) == 2

    def test_none_parent_id(self):
        """Root-level sections have no parent."""
        node = _make_node(parent_id=None, level=0, type="section", clause_number=None)
        assert node.parent_id is None

    def test_metadata_stored(self):
        node = _make_node(metadata={"paragraph_tag": "(1)P"})
        assert node.metadata["paragraph_tag"] == "(1)P"


class TestNodeSerialisation:
    def test_to_dict_round_trip(self):
        node = _make_node(cross_refs=["4.4.1.2"])
        d = node.to_dict()
        restored = Node.from_dict(d)
        assert restored.id == node.id
        assert restored.clause_number == node.clause_number
        assert restored.cross_refs == node.cross_refs
        assert restored.level == node.level

    def test_to_dict_is_json_serialisable(self):
        node = _make_node()
        d = node.to_dict()
        # Must not raise
        serialised = json.dumps(d, ensure_ascii=False)
        assert "6.2.2" in serialised

    def test_to_json(self):
        node = _make_node()
        j = node.to_json()
        parsed = json.loads(j)
        assert parsed["clause_number"] == "6.2.2"


class TestSaveLoadNodes:
    def test_save_and_load_round_trip(self, tmp_path):
        nodes = [
            _make_node(id="n1", clause_number="6.2.2"),
            _make_node(id="n2", clause_number="6.2.3", type="subclause"),
            _make_node(
                id="n3",
                clause_number=None,
                type="paragraph",
                parent_id="n1",
                level=3,
            ),
        ]
        path = str(tmp_path / "nodes.json")
        save_nodes(nodes, path)

        loaded = load_nodes(path)
        assert len(loaded) == 3
        assert loaded[0].id == "n1"
        assert loaded[1].clause_number == "6.2.3"
        assert loaded[2].parent_id == "n1"

    def test_save_preserves_cross_refs(self, tmp_path):
        nodes = [_make_node(cross_refs=["3.1.2", "4.4.1.2", "2.4.2.4"])]
        path = str(tmp_path / "nodes.json")
        save_nodes(nodes, path)
        loaded = load_nodes(path)
        assert loaded[0].cross_refs == ["3.1.2", "4.4.1.2", "2.4.2.4"]

    def test_save_empty_list(self, tmp_path):
        path = str(tmp_path / "empty.json")
        save_nodes([], path)
        loaded = load_nodes(path)
        assert loaded == []
