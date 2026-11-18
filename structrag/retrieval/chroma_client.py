"""
structrag/retrieval/chroma_client.py
-------------------------------------
Manages the local persistent ChromaDB instance for StructRAG.

Three collections are maintained:
  - ec2_hierarchical : clause-boundary chunks from hierarchical_chunker.py
  - ec2_flat         : sliding-window baseline from flat_chunker.py
  - ec2_child        : fine-grained child chunks from parent_child_chunker.py

Why ChromaDB?
- Local persistent storage — no server process required.
- Pluggable embedding functions — SentenceTransformer drops in directly.
- Metadata filtering — WHERE clauses let us filter by source_id, clause_number,
  level, etc. alongside semantic similarity.

Embedding model: all-MiniLM-L6-v2
- Fast (22 M params), good multilingual quality for technical French/English text.
- Consistent with the tokeniser used by the BM25 index in hybrid_retriever.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from chromadb.utils import embedding_functions

from structrag.config import (
    CHROMA_DIR,
    COLLECTION_CHILD,
    COLLECTION_FLAT,
    COLLECTION_HIERARCHICAL,
    EMBEDDING_MODEL,
)


# ---------------------------------------------------------------------------
# Re-export collection name constants so importers only need this module
# ---------------------------------------------------------------------------
__all__ = [
    "get_client",
    "get_collection",
    "list_collections",
    "delete_collection",
    "get_embedding_function",
    "COLLECTION_HIERARCHICAL",
    "COLLECTION_FLAT",
    "COLLECTION_CHILD",
]


def get_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Return a ChromaDB-compatible SentenceTransformer embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_client() -> chromadb.PersistentClient:
    """Return (or create) the persistent ChromaDB client at the configured path."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(
    name: str,
    create_if_missing: bool = True,
) -> chromadb.Collection:
    """
    Return a ChromaDB collection by name.

    Parameters
    ----------
    name               : collection name (use the COLLECTION_* constants)
    create_if_missing  : if True, create the collection on first access

    Returns
    -------
    chromadb.Collection
    """
    client = get_client()
    ef = get_embedding_function()

    if create_if_missing:
        return client.get_or_create_collection(
            name=name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=name, embedding_function=ef)


def list_collections() -> list[str]:
    """Return names of all existing collections."""
    client = get_client()
    return list(client.list_collections())   # v0.6+: returns names directly


def delete_collection(name: str) -> None:
    """Delete a collection (use when re-indexing from scratch)."""
    client = get_client()
    client.delete_collection(name)
    print(f"Deleted collection: {name}")
