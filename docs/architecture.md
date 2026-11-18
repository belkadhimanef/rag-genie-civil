# Architecture

## Overview

RAG Génie Civil is a research pipeline that evaluates whether respecting a document's native hierarchy during chunking produces meaningfully better retrieval quality compared with structure-blind fixed-window chunking.

The complete pipeline has six stages:

```
PDF document
    │
    ▼
┌─────────────────────────────────────┐
│  1. PARSING                         │
│  standards_parser.py                │
│  • Font-size + bold heading detect  │
│  • 2 100+ structured Node objects   │
│  • Cross-reference extraction       │
└──────────────────┬──────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
┌────────────────┐   ┌────────────────────┐
│ 2a. HIER.      │   │ 2b. FLAT BASELINE  │
│ CHUNKING       │   │ flat_chunker.py    │
│ hierarchical_  │   │ • Sliding 512-token│
│ chunker.py     │   │   window           │
│ • Clause-bound │   │ • No structure     │
│   chunks       │   │   awareness        │
│ • Ancestor path│   └────────┬───────────┘
│   prepended    │            │
└────────┬───────┘            │
         │                    │
         ▼                    ▼
┌─────────────────────────────────────┐
│  3. VECTOR INDEXING                 │
│  indexer.py + chroma_client.py      │
│  • all-MiniLM-L6-v2 embeddings      │
│  • ChromaDB persistent collections  │
│  • ec2_hierarchical | ec2_flat      │
│  • ec2_child (parent-child chunks)  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  4. RETRIEVAL (hybrid)              │
│  hybrid_retriever.py                │
│  • BM25 (rank-bm25, French stem)    │
│  • Vector cosine similarity         │
│  • Reciprocal Rank Fusion (k=60)    │
│                                     │
│  Enrichment layers:                 │
│  • query_rewriter.py  (G3)          │
│  • symbol_dictionary.py (G2)        │
│  • table_router.py                  │
│  • formula_library.py               │
│  • reranker.py (cross-encoder)      │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  5. GENERATION                      │
│  multi_model_generation.py          │
│  • Groq  (openai/gpt-oss-120b)      │
│  • Gemini (gemini-2.5-flash)        │
│  • NVIDIA (llama-3.1-70b)           │
│  • Structured citation-enforcing    │
│    system prompt                    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  6. EVALUATION                      │
│  hallucination_guard.py             │
│  score_all_models.py                │
│  • RAGAS: faithfulness, relevancy,  │
│    context precision, context recall│
│  • LLM-as-judge (llama-3.3-70b)     │
│  • Hallucination guard (clause check│
│    against 2 100+ known nodes)      │
└─────────────────────────────────────┘
```

## Module Map

### `structrag/parsing/`

| Module | Purpose |
|--------|---------|
| `hierarchy_schema.py` | `Node` dataclass — shared data model for all parsers |
| `standards_parser.py` | EC2 PDF → flat list of `Node` objects via PyMuPDF font-size detection |
| `table_extractor.py` | Extracts 5 key tables using pdfplumber (Tables 2.1N, 3.1, 4.1, 4.3N, 4.4N) |

### `structrag/chunking/`

| Module | Chunk type | Strategy |
|--------|-----------|---------|
| `hierarchical_chunker.py` | `hierarchical` | One chunk = complete clause + all direct children, ancestor path prepended |
| `flat_chunker.py` | `flat` | Fixed 512-token sliding window, no structure awareness |
| `parent_child_chunker.py` | `child` / `parent` | Fine-grained children indexed for retrieval; large parents used for generation context |

### `structrag/retrieval/`

| Module | Purpose |
|--------|---------|
| `chroma_client.py` | ChromaDB persistent client, collection management |
| `indexer.py` | Batch upsert of chunk JSON files into ChromaDB |
| `retriever.py` | `retrieve_hierarchical()`, `retrieve_flat()`, `retrieve_child()` |
| `hybrid_retriever.py` | BM25 + vector retrieval fused with RRF |
| `reranker.py` | Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) |
| `query_rewriter.py` | Symbol-aware query expansion (fast regex + optional LLM) |
| `symbol_dictionary.py` | EC2 symbol → formula/definition registry, injects into context |
| `table_router.py` | Direct numeric table lookups (Tables 2.1N, 3.1, 4.1, 4.3N, 4.4N) |
| `formula_library.py` | EC2 formula text library keyed by clause number |
| `inject_tables.py` | One-time script: injects pdfplumber tables into ChromaDB |

### `structrag/rag_pipeline/`

| Module | Purpose |
|--------|---------|
| `generation.py` | OpenAI chat completions backend + `GenerationResult` dataclass |
| `groq_generation.py` | Groq backend + domain classifier + rate-limit handling |
| `multi_model_generation.py` | Unified interface for Groq / Gemini / NVIDIA backends |
| `app.py` | CLI / REPL entry point (non-web) |

### `structrag/evaluation/`

| Module | Purpose |
|--------|---------|
| `hallucination_guard.py` | Clause-level citation verification against known node set |
| `qa_benchmark.py` | 50-question benchmark (40 FR + 10 EN) with ground truth |
| `ragas_runner.py` | RAGAS evaluation pipeline runner |
| `score_benchmark.py` | Scoring utilities |

## Configuration

All runtime configuration lives in `structrag/config.py`. Every path is derived from `Path(__file__).resolve().parent.parent` — no absolute paths appear anywhere else in the codebase.

Environment variables are loaded from `.env` automatically when `structrag.config` is imported. See `.env.example` for the full list of supported variables.
