# Evaluation

## Benchmark Dataset

A 50-question QA benchmark was constructed manually, covering the range of question types a practising structural engineer would ask:

| Language | Count |
|----------|-------|
| French   | 40    |
| English  | 10    |
| **Total**| **50**|

| Category        | Count | Description |
|-----------------|-------|-------------|
| `cover`         | 15    | Enrobage, exposure classes, structural classification |
| `materials`     | 10    | Concrete and steel properties, partial safety factors |
| `uls`           | 6     | Shear, bending, punching at ultimate limit state |
| `sls`           | 3     | Crack width, deflection, stress limits |
| `detailing`     | 7     | Anchorage, lap lengths, minimum reinforcement, bar spacing |
| `design_basis`  | 2     | Principles vs. application rules, load combinations |
| `analysis`      | 2     | Effective span, structural analysis methods |
| `cross_clause`  | 2     | Multi-clause questions requiring cross-reference |
| `out_of_scope`  | 5     | Questions outside EC2 (Eurocode 1, Eurocode 8, EN 206-1) |

Difficulty distribution: easy (15), medium (25), hard (10).

The 5 out-of-scope questions are included to measure the system's refusal behaviour. They are excluded from RAGAS scoring (marked `in_scope: false`).

## RAGAS Metrics

Scoring uses a single LLM-judge call per (model × question) pair (`meta/llama-3.3-70b-instruct` via NVIDIA build.nvidia.com), scoring four metrics simultaneously:

| Metric | Definition |
|--------|-----------|
| **Faithfulness** | Is the answer consistent with the retrieved context? Measures hallucination at the statement level (0 = contradicts context, 1 = fully grounded) |
| **Answer Relevancy** | Does the answer directly address the question? (0 = irrelevant, 1 = fully on-target) |
| **Context Precision** | Is the retrieved context relevant to the question? Measures retrieval precision (0 = all irrelevant, 1 = all relevant) |
| **Context Recall** | Does the retrieved context contain the information needed to construct the ground-truth answer? Measures retrieval coverage |

Using a single combined call per pair (rather than 4 separate calls) reduces API usage by 5× and eliminates prompt-ordering effects across metrics.

## Hallucination Guard

A separate `hallucination_guard.py` layer cross-checks every cited clause number in the model's answer against the set of known clause numbers extracted during parsing (~2,100 real nodes).

A clause is considered hallucinated if:
- It does not appear in the known node set, **and**
- Its immediate parent clause also does not appear in the known node set

This tolerates one-level sub-clause extensions (e.g. citing `6.2.2.1` when `6.2.2` is known) while catching fabricated numbers like `7.9.4.12`.

## Benchmark Results

The table below shows RAGAS scores from the full benchmark run (49 in-scope questions after deduplication). Results are stored in `structrag/evaluation/ragas_comparison.json`.

| Rank | Model | Provider | Faithfulness | Ans. Relevancy | Ctx Precision | Ctx Recall |
|------|-------|----------|:---:|:---:|:---:|:---:|
| 1 | `z-ai/glm-5.2` | NVIDIA | **0.682** | **0.636** | **0.682** | **0.632** |
| 2 | `openai/gpt-oss-120b` | Groq | 0.659 | 0.466 | 0.494 | 0.352 |
| 3 | `nvidia/llama-3.3-nemotron-super-49b-v1` | NVIDIA | 0.602 | 0.364 | 0.416 | 0.336 |
| 4 | `meta/llama-3.1-70b-instruct` | NVIDIA | 0.548 | 0.425 | 0.414 | 0.302 |

**Hallucination rate: 0%** across all 4 models. Every clause number cited in every answer was verified against the parsed node set.

> **Note on refusal rates.** `gpt-oss-120b` and both NVIDIA models show high refusal rates on borderline questions because the domain classifier filtered them before retrieval. `glm-5.2` answered more boundary questions, contributing to its higher scored-question count and consequently higher average scores. The refusal behaviour difference reflects classifier aggressiveness, not generation quality differences on clearly in-scope questions.

## 14-Question Diagnostic Harness

A smaller 14-question harness (`tests/rag_eval.py`) was used iteratively during development to track retrieval improvements:

| Version | Correct / 14 | Key change |
|---------|-------------|------------|
| v1 | 3 / 14 | k=5, vector only, no tables |
| v2 | 7 / 14 | k=8, BM25+vector hybrid, tables injected |
| v3 | 9 / 14 | Cross-encoder reranking, query rewriter, symbol dict |

The 5 remaining failures fall into two categories: questions whose answers require reading a table cell (now partially addressed by the table router), and questions about content that is explicitly outside EC2 (correctly rejected).

## Evaluation Limitations

- The LLM judge (`llama-3.3-70b`) shares training data with some of the evaluated models, introducing a potential bias toward answers that match its own generation style.
- Ground-truth answers were written by the project team, not by a certified structural engineer, and have not been independently validated.
- The benchmark does not test multi-turn conversations, follow-up questions, or complex multi-clause derivations requiring chained reasoning.
- Only Eurocode 2 is indexed. The ACI 318 placeholder in `source_manifest.json` has not been implemented.
