"""
score_all_models.py
--------------------
Batch RAGAS evaluation runner.

Scores all configured models on the full QA benchmark using a single
combined LLM-judge call per (model × question) pair — 4 metrics in one
prompt, cutting API calls by 5×.

Progress is saved after every scored question so the run is safe to
interrupt and resume.

Models scored
-------------
Configure the MODELS dict below.  Each entry maps a short key to the
JSON file produced by the per-model evaluation runner.  Only files that
exist in structrag/evaluation/ are loaded; missing files are skipped with
a warning.

Judge model
-----------
Uses meta/llama-3.3-70b-instruct via NVIDIA build.nvidia.com as the judge.
This is deliberately separate from the Groq RAG models to avoid rate-limit
conflicts and to provide an independent evaluation signal.

Usage
-----
    python score_all_models.py
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.config import (  # noqa: E402
    EC2_SOURCE_ID,
    EVAL_DIR,
    JUDGE_MODEL,
    PARENT_CHUNKS_PATH,
    PROGRESS_FILE,
    SLEEP_BETWEEN_CALLS,
    SLEEP_RATE_LIMIT,
)

N_RETRIEVE = 20
N_RERANK   = 8

# Models to score — key: short name, value: JSON filename in EVAL_DIR
# Update this dict to match the files actually present in structrag/evaluation/
MODELS: dict[str, str] = {
    "gpt120b":     "ragas_gpt120b.json",
    "nemotron49b": "ragas_nemotron49b.json",
    "llama70b":    "ragas_llama70b.json",
    "glm52":       "ragas_glm52.json",
}

ERROR_PREFIXES = (
    "[ERROR", "[Groq rate", "[NVIDIA rate", "[Gemini",
    "[WRONGLY REJECTED", "[CORRECTLY REJECTED", "[No match",
    "[No chunk", "[No relev", "[Groq max",
)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def call_judge(prompt: str) -> str:
    """Call the judge model (NVIDIA build.nvidia.com)."""
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    for attempt in range(7):
        try:
            from openai import OpenAI
            resp = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
                timeout=60.0,
            ).chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = min(SLEEP_RATE_LIMIT * (attempt + 1), 210)
                print(f" [RL {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                print(f" [ERR {err[:40]}]", end="", flush=True)
                return ""
    return ""


def score_one(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, float]:
    """Score one (question, answer, context, ground_truth) tuple — 4 metrics in one call."""
    ctx_str = " | ".join(c[:120] for c in contexts[:3]) or "[no context]"

    prompt = (
        "You are evaluating a RAG system answer about Eurocode 2 (French civil "
        "engineering standard).\n"
        "The answer may be in French — this is normal and correct.\n\n"
        f"QUESTION: {question[:200]}\n\n"
        f"CONTEXT (retrieved passages): {ctx_str[:400]}\n\n"
        f"GROUND TRUTH: {ground_truth[:200]}\n\n"
        f"ANSWER: {answer[:300]}\n\n"
        "Score each metric from 0.0 to 1.0:\n"
        "1. FAITHFULNESS: Is the answer consistent with the context? "
        "(0=contradicts, 1=fully grounded)\n"
        "2. RELEVANCY: Does the answer directly address the question? "
        "(0=irrelevant, 1=fully answers)\n"
        "3. CONTEXT_PRECISION: Is the retrieved context relevant to the question? "
        "(0=irrelevant, 1=all relevant)\n"
        "4. CONTEXT_RECALL: Does the context contain info needed for the ground truth? "
        "(0=missing, 1=fully covered)\n\n"
        "Respond ONLY with these 4 lines (nothing else):\n"
        "FAITHFULNESS: X.X\n"
        "RELEVANCY: X.X\n"
        "CONTEXT_PRECISION: X.X\n"
        "CONTEXT_RECALL: X.X"
    )

    response = call_judge(prompt)
    time.sleep(SLEEP_BETWEEN_CALLS)

    def parse(label: str) -> float:
        m = re.search(rf'{label}:\s*(0\.\d+|1\.0|0\.0|[01])', response, re.IGNORECASE)
        if m:
            return min(1.0, max(0.0, float(m.group(1))))
        lines = [ln for ln in response.splitlines() if label.lower() in ln.lower()]
        if lines:
            nums = re.findall(r'0\.\d+|1\.0|0\.0', lines[0])
            if nums:
                return float(nums[0])
        return 0.5

    if not response:
        return {"f": 0.5, "ar": 0.5, "cp": 0.5, "cr": 0.5}

    return {
        "f":  parse("FAITHFULNESS"),
        "ar": parse("RELEVANCY"),
        "cp": parse("CONTEXT_PRECISION"),
        "cr": parse("CONTEXT_RECALL"),
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(question: str) -> list[str]:
    """Run the full hybrid retrieval pipeline and return top-5 chunk texts."""
    try:
        from structrag.chunking.parent_child_chunker import resolve_parents
        from structrag.retrieval.formula_library import get_formula_for_query, make_formula_chunk
        from structrag.retrieval.hybrid_retriever import retrieve_hybrid
        from structrag.retrieval.reranker import rerank_with_fallback
        from structrag.retrieval.retriever import retrieve_child
        from structrag.retrieval.table_router import enrich_with_table_router

        children = retrieve_child(question, n_results=N_RETRIEVE, source_id=EC2_SOURCE_ID)
        relevant = [c for c in children if c.get("score", 999) < 0.8]
        chunks = (
            resolve_parents(relevant, str(PARENT_CHUNKS_PATH))
            if relevant and PARENT_CHUNKS_PATH.exists()
            else []
        )

        hybrid = retrieve_hybrid(question, n_results=N_RETRIEVE, source_id=EC2_SOURCE_ID)
        seen = {c["chunk_id"] for c in chunks}
        for h in hybrid:
            if h["chunk_id"] not in seen:
                chunks.append(h)
                seen.add(h["chunk_id"])

        chunks = rerank_with_fallback(question, chunks, top_n=N_RERANK)
        chunks = enrich_with_table_router(question, chunks, EC2_SOURCE_ID)

        formula = get_formula_for_query(question)
        if formula:
            m = re.search(r'\b(\d+\.\d+(?:\.\d+)*)\b', formula)
            fc = make_formula_chunk(formula, m.group(1) if m else "unknown", EC2_SOURCE_ID)
            if fc["chunk_id"] not in {c["chunk_id"] for c in chunks}:
                chunks = [fc] + chunks

        return [c["text"] for c in chunks[:5]]
    except Exception as e:
        print(f"\n  [retrieval error: {e}]")
        return []


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {}


def save_progress(prog: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(prog, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load model JSONs that exist on disk
    model_data: dict[str, dict] = {}
    for key, fname in MODELS.items():
        path = EVAL_DIR / fname
        if path.exists():
            model_data[key] = json.loads(path.read_text(encoding="utf-8"))
            print(f"Loaded {key}: {len(model_data[key]['per_question'])} questions")
        else:
            print(f"MISSING (skipped): {fname}")

    if not model_data:
        print("No model JSON files found in structrag/evaluation/. Nothing to score.")
        return

    first     = next(iter(model_data.values()))
    questions = first["per_question"]
    n         = len(questions)
    progress  = load_progress()

    in_scope_idx = [i for i, q in enumerate(questions) if q.get("in_scope", True)]
    print(f"\nProgress  : {len(progress)} contexts already saved")
    print(f"In-scope  : {len(in_scope_idx)} of {n} questions")

    # Phase 1: retrieve context (re-use saved contexts)
    needs_retrieval = [i for i in in_scope_idx if str(i) not in progress]
    if needs_retrieval:
        print(f"\nPhase 1: Retrieving context for {len(needs_retrieval)} questions...")
        for i in needs_retrieval:
            q_text = questions[i]["question"]
            print(f"  Q{i + 1}/{n}: {q_text[:60]}...", end=" ", flush=True)
            ctx = retrieve(q_text)
            progress[str(i)] = {"context": ctx, "scores": {}}
            save_progress(progress)
            print(f"({len(ctx)} chunks)")
            time.sleep(0.3)
        print("  Retrieval complete.\n")
    else:
        print("Phase 1: All contexts already retrieved.\n")

    # Phase 2: score with judge
    todo = [
        (i, key)
        for i in in_scope_idx
        for key in model_data
        if key not in progress.get(str(i), {}).get("scores", {})
    ]

    print(f"Phase 2: {len(todo)} (model × question) pairs to score")

    for idx, (i, key) in enumerate(todo):
        q_item = questions[i]
        q_text = q_item["question"]
        gt     = q_item["ground_truth"]
        ctx    = progress[str(i)]["context"] or ["[no context]"]

        per_q  = model_data[key]["per_question"]
        answer = per_q[i]["answer"] if i < len(per_q) else ""
        is_err = not answer or any(answer.startswith(p) for p in ERROR_PREFIXES)

        print(f"  [{idx + 1}/{len(todo)}] Q{i + 1} [{key}]: {q_text[:50]}...",
              end=" ", flush=True)

        if is_err:
            scores: dict[str, float] = {"f": 0.0, "ar": 0.0, "cp": 0.0, "cr": 0.0}
            print("SKIP (error answer)")
        else:
            scores = score_one(q_text, answer, ctx, gt)
            print(f"f={scores['f']:.2f} ar={scores['ar']:.2f} "
                  f"cp={scores['cp']:.2f} cr={scores['cr']:.2f}")

        progress[str(i)]["scores"][key] = scores
        save_progress(progress)

    # Phase 3: write updated JSONs and print summary table
    print("\n\nPhase 3: Writing results ...")

    def avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    for key, data in model_data.items():
        per_q = data["per_question"]
        scored: list[dict] = []

        for i, q_item in enumerate(per_q):
            if not q_item.get("in_scope", True):
                continue
            s = progress.get(str(i), {}).get("scores", {}).get(key)
            if s:
                q_item.update({
                    "faithfulness":      s["f"],
                    "answer_relevancy":  s["ar"],
                    "context_precision": s["cp"],
                    "context_recall":    s["cr"],
                    "context_texts":     [c[:500] for c in progress[str(i)]["context"][:5]],
                })
                scored.append(q_item)

        data["ragas_scores"] = {
            "faithfulness":      avg([q["faithfulness"]      for q in scored]),
            "answer_relevancy":  avg([q["answer_relevancy"]  for q in scored]),
            "context_precision": avg([q["context_precision"] for q in scored]),
            "context_recall":    avg([q["context_recall"]    for q in scored]),
        }

        (EVAL_DIR / MODELS[key]).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Comparison JSON
    comparison = {
        k: json.loads((EVAL_DIR / v).read_text(encoding="utf-8"))
        for k, v in MODELS.items()
        if (EVAL_DIR / v).exists()
    }
    (EVAL_DIR / "ragas_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Summary table
    print(f"\n{'='*82}")
    print("  FINAL RAGAS SCORES")
    print(f"{'='*82}")
    print(f"  {'Model':<32} {'Faith':>7} {'AnswRel':>8} {'CtxPrec':>9} {'CtxRec':>8}")
    print("  " + "-" * 68)
    for key, data in comparison.items():
        s   = data.get("ragas_scores", {})
        mid = data.get("model_id", key)
        print(
            f"  {mid[:32]:<32} "
            f"{s.get('faithfulness', 0):>7.4f} "
            f"{s.get('answer_relevancy', 0):>8.4f} "
            f"{s.get('context_precision', 0):>9.4f} "
            f"{s.get('context_recall', 0):>8.4f}"
        )
    print(f"\nSaved: ragas_comparison.json  |  Progress: {PROGRESS_FILE.name}")


if __name__ == "__main__":
    main()
