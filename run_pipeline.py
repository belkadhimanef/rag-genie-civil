"""
run_pipeline.py
---------------
Master pipeline runner for RAG Génie Civil.

Executes the full ingestion pipeline in sequence:

  Step 1  Parse EC2 PDF          -> data/parsed/ec2_2004_nf_nodes.json
  Step 2  Hierarchical chunks    -> data/chunks/ec2_hierarchical_chunks.json
  Step 3  Flat chunks (baseline) -> data/chunks/ec2_flat_chunks.json
  Step 4  Parent-child chunks    -> data/chunks/ec2_{child,parent}_chunks.json
  Step 5  Index into ChromaDB    -> data/chroma_db/
  Step 6  Smoke test (retrieval only — no LLM required)

Usage
-----
    # Full run from scratch
    python run_pipeline.py

    # Resume from step 2 (re-use existing parsed nodes)
    python run_pipeline.py --from 2

    # Re-index only (chunks already built)
    python run_pipeline.py --from 5

    # Print what steps exist and exit
    python run_pipeline.py --list

Requirements
------------
  - Place the EC2 PDF in the project root directory.
  - Set GROQ_API_KEY (or other LLM keys) in a .env file.
    The pipeline itself does not call any LLM — only the smoke test
    queries retrieval, so no API key is needed to run steps 1-5.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 crash with French characters)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Project root is the directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Config must be imported after sys.path is set
from structrag.config import (  # noqa: E402
    CHILD_CHUNKS_PATH,
    CHUNKS_DIR,
    EC2_SOURCE_ID,
    FLAT_CHUNKS_PATH,
    HIER_CHUNKS_PATH,
    NODES_PATH,
    PARENT_CHUNKS_PATH,
    PARSED_DIR,
)

# ---------------------------------------------------------------------------
# Locate the EC2 PDF automatically
# ---------------------------------------------------------------------------

def _find_pdf() -> Path:
    """
    Look for a PDF file in the project root.
    Tries the known AFNOR filename first, then falls back to any *.pdf.
    Raises FileNotFoundError with a helpful message if none is found.
    """
    known = PROJECT_ROOT / "FA039724-NF EN 1992-1-1 (octobre 2005) Eurocode 2.pdf"
    if known.exists():
        return known
    pdfs = list(PROJECT_ROOT.glob("*.pdf"))
    if pdfs:
        return pdfs[0]
    raise FileNotFoundError(
        "No PDF found in the project root.\n"
        "Place the Eurocode 2 PDF (NF EN 1992-1-1:2004) in the project root "
        "and re-run the pipeline.\n"
        "The file must not be committed to the repository (see .gitignore)."
    )


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

_log_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _log_lines.append(str(msg))


def save_log(path: Path) -> None:
    path.write_text("\n".join(_log_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 1 — Parse
# ---------------------------------------------------------------------------

def step_parse(pdf_path: Path) -> None:
    log("\n=== STEP 1: Parsing EC2 PDF ===")
    from structrag.parsing.hierarchy_schema import save_nodes
    from structrag.parsing.standards_parser import EC2_CONFIG, extract_cross_refs, parse_standard

    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    nodes = parse_standard(str(pdf_path), EC2_CONFIG)
    nodes = extract_cross_refs(nodes)
    save_nodes(nodes, str(NODES_PATH))

    counts = Counter(n.type for n in nodes)
    log(f"Parsed {len(nodes)} nodes in {time.time() - t0:.1f}s")
    for node_type, count in sorted(counts.items()):
        log(f"  {node_type:12s}  {count}")


# ---------------------------------------------------------------------------
# Step 2 — Hierarchical chunking
# ---------------------------------------------------------------------------

def step_chunk_hierarchical() -> None:
    log("\n=== STEP 2: Hierarchical chunking ===")
    from structrag.chunking.hierarchical_chunker import chunk_hierarchical, save_chunks
    from structrag.parsing.hierarchy_schema import load_nodes

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    nodes = load_nodes(str(NODES_PATH))
    chunks = chunk_hierarchical(nodes, source_id=EC2_SOURCE_ID)
    save_chunks(chunks, str(HIER_CHUNKS_PATH))

    tokens = [c["token_count"] for c in chunks]
    log(f"Built {len(chunks)} hierarchical chunks in {time.time() - t0:.1f}s")
    log(f"  Token stats — min:{min(tokens)}  "
        f"avg:{sum(tokens)//len(tokens)}  max:{max(tokens)}")
    multi = sum(1 for c in chunks if c["part_index"] > 0)
    log(f"  Continuation parts (split long clauses): {multi}")


# ---------------------------------------------------------------------------
# Step 3 — Flat chunking
# ---------------------------------------------------------------------------

def step_chunk_flat(pdf_path: Path) -> None:
    log("\n=== STEP 3: Flat chunking (baseline) ===")
    from structrag.chunking.flat_chunker import _extract_pages, _sliding_window_chunks, save_chunks

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pages = _extract_pages(str(pdf_path))
    chunks = _sliding_window_chunks(pages, source_id=EC2_SOURCE_ID)
    save_chunks(chunks, str(FLAT_CHUNKS_PATH))

    tokens = [c["token_count"] for c in chunks]
    log(f"Built {len(chunks)} flat chunks in {time.time() - t0:.1f}s")
    log(f"  Token stats — min:{min(tokens)}  "
        f"avg:{sum(tokens)//len(tokens)}  max:{max(tokens)}")


# ---------------------------------------------------------------------------
# Step 4 — Parent-child chunking
# ---------------------------------------------------------------------------

def step_chunk_parent_child(pdf_path: Path) -> None:
    log("\n=== STEP 4: Parent-child chunking ===")
    from structrag.chunking.parent_child_chunker import (
        _add_table_nodes_to_parent_child,
        chunk_parent_child,
        save_chunks,
    )
    from structrag.parsing.hierarchy_schema import load_nodes

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    nodes = load_nodes(str(NODES_PATH))
    child_chunks, parent_chunks = chunk_parent_child(nodes, source_id=EC2_SOURCE_ID)
    child_chunks, parent_chunks = _add_table_nodes_to_parent_child(
        str(pdf_path), EC2_SOURCE_ID, child_chunks, parent_chunks
    )
    save_chunks(child_chunks,  str(CHILD_CHUNKS_PATH))
    save_chunks(parent_chunks, str(PARENT_CHUNKS_PATH))

    log(f"Built {len(parent_chunks)} parent + {len(child_chunks)} child chunks "
        f"in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Step 5 — Index into ChromaDB
# ---------------------------------------------------------------------------

def step_index() -> None:
    log("\n=== STEP 5: Indexing into ChromaDB ===")
    from structrag.retrieval.chroma_client import (
        COLLECTION_CHILD,
        COLLECTION_FLAT,
        COLLECTION_HIERARCHICAL,
        delete_collection,
        get_collection,
        list_collections,
    )
    from structrag.retrieval.indexer import index_child_chunks, index_flat, index_hierarchical

    # Clear existing collections to avoid stale data on re-runs
    existing = list_collections()
    for name in [COLLECTION_HIERARCHICAL, COLLECTION_FLAT, COLLECTION_CHILD]:
        if name in existing:
            log(f"  Clearing existing collection: {name}")
            delete_collection(name)

    t0 = time.time()
    index_hierarchical(str(HIER_CHUNKS_PATH))
    index_flat(str(FLAT_CHUNKS_PATH))

    if CHILD_CHUNKS_PATH.exists():
        index_child_chunks(str(CHILD_CHUNKS_PATH))
    else:
        log("  Skipping child chunks (file not found — run step 4 to build them)")

    log(f"\nIndexing completed in {time.time() - t0:.1f}s")

    log("\nCollections in ChromaDB:")
    for name in list_collections():
        col = get_collection(name, create_if_missing=False)
        log(f"  {name}  ->  {col.count()} documents")


# ---------------------------------------------------------------------------
# Step 6 — Smoke test
# ---------------------------------------------------------------------------

def step_smoke_test() -> None:
    log("\n=== STEP 6: Smoke test (retrieval only — no LLM) ===")
    from structrag.retrieval.retriever import retrieve_flat, retrieve_hierarchical

    test_queries = [
        "What is the minimum concrete cover for a beam in exposure class XC2?",
        "Shear resistance formula for members without shear reinforcement",
        "Minimum reinforcement ratio for bending",
    ]

    for query in test_queries:
        log(f"\nQuery: {query}")

        hier = retrieve_hierarchical(query, n_results=3, source_id=EC2_SOURCE_ID)
        log(f"  Hierarchical top-3:")
        for r in hier:
            log(f"    score={r['score']}  clause={r.get('clause_number', '')}  "
                f"p={r.get('page_number', '')}  "
                f"path={r.get('ancestor_path', '')[:60]}")

        flat = retrieve_flat(query, n_results=3, source_id=EC2_SOURCE_ID)
        log(f"  Flat top-3:")
        for r in flat:
            log(f"    score={r['score']}  pages={r.get('start_page', '')}-{r.get('end_page', '')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEP_DESCRIPTIONS = {
    1: "Parse EC2 PDF → ec2_2004_nf_nodes.json",
    2: "Hierarchical chunking → ec2_hierarchical_chunks.json",
    3: "Flat chunking (baseline) → ec2_flat_chunks.json",
    4: "Parent-child chunking → ec2_{child,parent}_chunks.json",
    5: "Index into ChromaDB",
    6: "Smoke test (retrieval)",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Génie Civil — pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1,
        help="Start from step N (default: 1)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List pipeline steps and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Pipeline steps:")
        for n, desc in STEP_DESCRIPTIONS.items():
            print(f"  {n}. {desc}")
        sys.exit(0)

    start = args.from_step

    try:
        pdf_path = _find_pdf()
        log(f"PDF: {pdf_path.name}")

        if start <= 1:
            step_parse(pdf_path)
        if start <= 2:
            step_chunk_hierarchical()
        if start <= 3:
            step_chunk_flat(pdf_path)
        if start <= 4:
            step_chunk_parent_child(pdf_path)
        if start <= 5:
            step_index()
        if start <= 6:
            step_smoke_test()

        log(f"\n{'='*60}")
        log("Pipeline complete.")
        log(f"{'='*60}")
        log("\nNext steps:")
        log("  Web interface:  python app.py")
        log("  CLI interface:  python structrag/rag_pipeline/app.py --groq")
        log("  Evaluation:     python score_all_models.py")

    except FileNotFoundError as exc:
        log(f"\nERROR: {exc}")
        sys.exit(1)
    except Exception:
        import traceback
        log(f"\nPipeline failed:")
        log(traceback.format_exc())
        sys.exit(1)
    finally:
        log_path = PROJECT_ROOT / "pipeline_run.log"
        save_log(log_path)
        print(f"\nLog saved -> {log_path}")
