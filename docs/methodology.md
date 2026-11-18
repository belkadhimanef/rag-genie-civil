# Methodology

## Research Question

Can respecting a technical standard's native document hierarchy during chunking produce measurably better retrieval quality compared with conventional fixed-window chunking, when the retrieval task involves precise clause-level lookups in a dense, cross-referential normative document?

## Document Characteristics

Eurocode 2 (NF EN 1992-1-1:2004) is a 211-page normative standard with the following structural properties that make it a challenging retrieval corpus:

- **Deep hierarchy**: sections → clauses → subclauses → paragraphs → notes, up to 4 levels deep
- **Conditional structure**: a paragraph like `(1)P` often states a principle that only makes sense together with its applicability note two paragraphs later
- **Formula-note coupling**: formulas appear as image blocks; the defining conditions and national annex values appear in `NOTE` paragraphs immediately after
- **Dense cross-referencing**: approximately 569 cross-references were extracted automatically — clauses frequently point to tables and other clauses for the full picture
- **Bilingual technical vocabulary**: French document body with Greek/Latin symbols; queries arrive in both French and English

## Chunking Strategies

### Strategy 1 — Flat (Baseline)

```
Full PDF text → strip headers → concatenate pages → tokenise → slide 512-token window
```

- Window size: 512 tokens, 64-token overlap
- No awareness of clause numbers, headings, or structural level
- A single window routinely contains fragments from two or three different subclauses
- No ancestor path or clause identifier stored with the chunk

This is the standard approach used in most general-purpose RAG tutorials and frameworks.

### Strategy 2 — Hierarchical

```
Parsed nodes → group by heading node → assemble heading + all direct children
```

- One chunk per subclause (or clause for shallow clauses without subclauses)
- Text begins with `[ancestor path]` so the embedding captures structural location
- All `(N)P` paragraphs, `NOTE` blocks, and table placeholders that belong to a clause are assembled into one chunk
- If a clause exceeds 512 tokens, it is split at paragraph boundaries — each part repeats the heading block so every chunk is self-contained

### Strategy 3 — Parent-Child (Small-to-Big)

```
Parsed nodes → fine-grained child chunks (≤200 tokens) for retrieval
             → full-clause parent chunks (≤1500 tokens) for generation
```

- **Child chunks**: one paragraph or one table row — indexed in ChromaDB
- **Parent chunks**: the complete clause text — looked up by `parent_chunk_id` after retrieval
- Retrieval is precise (small embedding signal); generation context is complete (full clause)
- Table rows are split individually with the caption + header repeated for self-containment

## Retrieval Pipeline

The production retrieval pipeline uses all three strategies in combination:

1. **Child vector search** — retrieve top-20 child chunks by cosine similarity
2. **Parent lookup** — resolve each child to its parent chunk (de-duplicated)
3. **Hybrid BM25 + vector** — run separately on the hierarchical collection, fuse with RRF (k=60)
4. **Merge and de-duplicate** — combine parent-resolved chunks and hybrid results
5. **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` scores (query, chunk) pairs and reorders
6. **Enrichment** — inject relevant table lookups, symbol definitions, and formula text as additional context chunks

## Embedding Model

`all-MiniLM-L6-v2` (22M parameters, multilingual). Selected for:
- Reasonable French text encoding quality despite not being a dedicated French model
- Fast inference (suitable for real-time retrieval)
- Compatibility with ChromaDB's built-in embedding function interface

## Document Parsing

The EC2 PDF is born-digital (not scanned), produced by iText 1.4.1. Heading detection uses font size and bold flags extracted via PyMuPDF span metadata:

| Font size | Bold | Classification |
|-----------|------|---------------|
| ≥ 12 pt   | yes  | Section heading |
| 11 pt     | yes  | Clause (level 1) |
| 10 pt     | yes  | Subclause (level 2+) |
| 10 pt     | no   | Body paragraph |
| ≤ 9 pt    | —    | Note or subscript |

A lightweight mojibake correction pass handles the Latin-1/UTF-8 encoding artefacts in the AFNOR edition (characters like `Ú → é`, `Ó → à`, etc.).

## Evaluation Protocol

See [`evaluation.md`](evaluation.md) for the full RAGAS evaluation methodology and results.
