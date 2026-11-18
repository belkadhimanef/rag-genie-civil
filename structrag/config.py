"""
structrag/config.py
-------------------
Central configuration for StructRAG.

All paths are derived from the project root using Path(__file__), so the
project is fully portable — no hardcoded absolute paths anywhere in the
codebase.  Every other module imports from here instead of constructing
paths locally.

Environment variables are loaded from .env at project root when this module
is imported.  Downstream modules can read os.environ directly.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — two levels up from this file (structrag/config.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Load .env (python-dotenv preferred; plain fallback for environments without it)
# ---------------------------------------------------------------------------
_env_path = PROJECT_ROOT / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(_env_path, override=False)
except ImportError:
    # Fallback: manual parse (no dependency required)
    if _env_path.exists():
        for _line in _env_path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _k, _v = _k.strip(), _v.strip()
                if _k not in os.environ:          # don't override shell exports
                    os.environ[_k] = _v

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR:   Path = PROJECT_ROOT / "structrag" / "data"
PARSED_DIR: Path = DATA_DIR / "parsed"
CHUNKS_DIR: Path = DATA_DIR / "chunks"
CHROMA_DIR: Path = DATA_DIR / "chroma_db"

# ---------------------------------------------------------------------------
# Well-known file paths
# ---------------------------------------------------------------------------
NODES_PATH:        Path = PARSED_DIR / "ec2_2004_nf_nodes.json"
HIER_CHUNKS_PATH:  Path = CHUNKS_DIR / "ec2_hierarchical_chunks.json"
FLAT_CHUNKS_PATH:  Path = CHUNKS_DIR / "ec2_flat_chunks.json"
CHILD_CHUNKS_PATH: Path = CHUNKS_DIR / "ec2_child_chunks.json"
PARENT_CHUNKS_PATH: Path = CHUNKS_DIR / "ec2_parent_chunks.json"
SOURCE_MANIFEST:   Path = DATA_DIR   / "source_manifest.json"

# ---------------------------------------------------------------------------
# ChromaDB collection names
# ---------------------------------------------------------------------------
COLLECTION_HIERARCHICAL: str = "ec2_hierarchical"
COLLECTION_FLAT:         str = "ec2_flat"
COLLECTION_CHILD:        str = "ec2_child"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------
HIER_MAX_TOKENS:     int = 512
HIER_OVERLAP_TOKENS: int = 64
FLAT_MAX_TOKENS:     int = 512
FLAT_OVERLAP_TOKENS: int = 64
PARENT_MAX_TOKENS:   int = 1500
CHILD_MAX_TOKENS:    int = 200
FLAT_SKIP_PAGES:     int = 17   # cover + TOC pages (0-indexed)

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
DEFAULT_N_RESULTS:    int   = 5
RERANK_TOP_N:         int   = 8
RELEVANCE_THRESHOLD:  float = 1.2   # cosine distance cutoff
RRF_K:                int   = 60    # Reciprocal Rank Fusion constant

# ---------------------------------------------------------------------------
# LLM defaults (overridden by environment variables)
# ---------------------------------------------------------------------------
DEFAULT_GROQ_MODEL:   str = os.environ.get("GROQ_MODEL",   "openai/gpt-oss-120b")
DEFAULT_GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_NVIDIA_MODEL: str = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")

DEFAULT_MAX_TOKENS:  int   = 1024
DEFAULT_TEMPERATURE: float = 0.0

# ---------------------------------------------------------------------------
# Source identifier
# ---------------------------------------------------------------------------
EC2_SOURCE_ID: str = "ec2_2004_nf"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_DIR:           Path = PROJECT_ROOT / "structrag" / "evaluation"
PROGRESS_FILE:      Path = EVAL_DIR / "scoring_progress.json"
JUDGE_MODEL:        str  = "meta/llama-3.3-70b-instruct"
SLEEP_BETWEEN_CALLS: int = 3    # seconds between successful judge calls
SLEEP_RATE_LIMIT:    int = 30   # base wait on 429 responses
