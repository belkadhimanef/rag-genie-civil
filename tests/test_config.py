"""
tests/test_config.py
--------------------
Tests for central configuration module.

These tests do not require a PDF, API keys, or a running ChromaDB instance.
They verify that paths are correctly derived from the project root and that
constants have sensible values.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_project_root_exists():
    """PROJECT_ROOT must point to an existing directory."""
    from structrag.config import PROJECT_ROOT
    assert PROJECT_ROOT.exists(), f"PROJECT_ROOT does not exist: {PROJECT_ROOT}"
    assert PROJECT_ROOT.is_dir()


def test_project_root_contains_expected_files():
    """PROJECT_ROOT should contain the key project files."""
    from structrag.config import PROJECT_ROOT
    assert (PROJECT_ROOT / "structrag").is_dir()
    assert (PROJECT_ROOT / "requirements.txt").exists()


def test_data_paths_are_under_project_root():
    """All data paths must be children of PROJECT_ROOT."""
    from structrag.config import (
        CHROMA_DIR,
        CHUNKS_DIR,
        DATA_DIR,
        PARSED_DIR,
        PROJECT_ROOT,
    )
    for path in [DATA_DIR, PARSED_DIR, CHUNKS_DIR, CHROMA_DIR]:
        assert str(path).startswith(str(PROJECT_ROOT)), (
            f"{path} is not under PROJECT_ROOT"
        )


def test_collection_name_constants():
    """ChromaDB collection names must be non-empty strings."""
    from structrag.config import (
        COLLECTION_CHILD,
        COLLECTION_FLAT,
        COLLECTION_HIERARCHICAL,
    )
    for name in [COLLECTION_HIERARCHICAL, COLLECTION_FLAT, COLLECTION_CHILD]:
        assert isinstance(name, str) and len(name) > 0


def test_chunking_parameters_sane():
    """Chunking token limits must be positive and logically consistent."""
    from structrag.config import (
        CHILD_MAX_TOKENS,
        FLAT_MAX_TOKENS,
        HIER_MAX_TOKENS,
        PARENT_MAX_TOKENS,
    )
    assert HIER_MAX_TOKENS > 0
    assert FLAT_MAX_TOKENS > 0
    assert CHILD_MAX_TOKENS > 0
    assert PARENT_MAX_TOKENS > CHILD_MAX_TOKENS, (
        "Parent chunks must be larger than child chunks"
    )


def test_embedding_model_name():
    """Embedding model name must be a non-empty string."""
    from structrag.config import EMBEDDING_MODEL
    assert isinstance(EMBEDDING_MODEL, str) and len(EMBEDDING_MODEL) > 0


def test_relevance_threshold_range():
    """Relevance threshold is a cosine distance in [0, 2]."""
    from structrag.config import RELEVANCE_THRESHOLD
    assert 0.0 < RELEVANCE_THRESHOLD <= 2.0


def test_ec2_source_id():
    """Source ID for EC2 must match the value used in chunk files."""
    from structrag.config import EC2_SOURCE_ID
    assert EC2_SOURCE_ID == "ec2_2004_nf"
