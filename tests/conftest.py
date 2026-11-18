"""
tests/conftest.py
-----------------
Shared pytest configuration and fixtures.

Markers defined here:
  - integration : tests that require a live ChromaDB instance or indexed data
  - llm         : tests that require a live API key (Groq, Gemini, NVIDIA)
  - pdf         : tests that require the EC2 PDF to be present in the project root
"""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires a running ChromaDB instance with indexed data",
    )
    config.addinivalue_line(
        "markers",
        "llm: requires a valid API key (GROQ_API_KEY, GEMINI_API_KEY, etc.)",
    )
    config.addinivalue_line(
        "markers",
        "pdf: requires the EC2 PDF to be present in the project root",
    )


@pytest.fixture(scope="session")
def project_root():
    from structrag.config import PROJECT_ROOT
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def pdf_path(project_root):
    """
    Returns the PDF path if the EC2 PDF exists, otherwise skips the test.
    Use with @pytest.mark.pdf tests.
    """
    candidates = list(project_root.glob("*.pdf"))
    if not candidates:
        pytest.skip("EC2 PDF not found in project root — skipping PDF-dependent test")
    return candidates[0]


@pytest.fixture(scope="session")
def nodes_path(project_root):
    """
    Returns the parsed nodes path if it exists, otherwise skips.
    Use for tests that need pre-parsed data.
    """
    from structrag.config import NODES_PATH
    if not NODES_PATH.exists():
        pytest.skip(
            f"Parsed nodes not found at {NODES_PATH}. "
            "Run `python run_pipeline.py --from 1` first."
        )
    return NODES_PATH
