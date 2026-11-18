"""
app.py
------
RAG Génie Civil — Flask web interface.

Serves a single-page chat UI backed by the full RAG pipeline:
  query rewriting → hybrid retrieval → cross-encoder reranking →
  table/formula injection → multi-model LLM generation

The LLM backend and model are configured via environment variables
(see .env.example).  Default: Groq / openai/gpt-oss-120b.

Usage
-----
    python app.py
    # Then open http://127.0.0.1:5000

Environment variables required
-------------------------------
    GROQ_API_KEY    — Groq API key
    GROQ_MODEL      — optional override (default: openai/gpt-oss-120b)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Config is imported first — it loads .env automatically
from structrag.config import EC2_SOURCE_ID, DEFAULT_GROQ_MODEL, PARENT_CHUNKS_PATH  # noqa: E402

from flask import Flask, jsonify, render_template_string, request  # noqa: E402

app = Flask(__name__)

MODEL_ID  = DEFAULT_GROQ_MODEL
PROVIDER  = "groq"

# ---------------------------------------------------------------------------
# Lazy pipeline initialisation
# ---------------------------------------------------------------------------

_pipeline_ready = False


def _ensure_pipeline() -> None:
    global _pipeline_ready
    if _pipeline_ready:
        return
    # Import all retrieval modules to trigger lazy index builds
    from structrag.retrieval.hybrid_retriever import retrieve_hybrid        # noqa: F401
    from structrag.retrieval.retriever import retrieve_child                # noqa: F401
    from structrag.chunking.parent_child_chunker import resolve_parents     # noqa: F401
    from structrag.retrieval.reranker import rerank_with_fallback           # noqa: F401
    from structrag.retrieval.table_router import enrich_with_table_router   # noqa: F401
    from structrag.retrieval.formula_library import (                       # noqa: F401
        get_formula_for_query, make_formula_chunk,
    )
    _pipeline_ready = True


# ---------------------------------------------------------------------------
# Retrieval pipeline
# ---------------------------------------------------------------------------

def retrieve(question: str) -> list[dict]:
    """Run the full retrieval pipeline for one question."""
    from structrag.chunking.parent_child_chunker import resolve_parents
    from structrag.retrieval.formula_library import get_formula_for_query, make_formula_chunk
    from structrag.retrieval.hybrid_retriever import retrieve_hybrid
    from structrag.retrieval.query_rewriter import rewrite_query
    from structrag.retrieval.reranker import rerank_with_fallback
    from structrag.retrieval.retriever import retrieve_child
    from structrag.retrieval.symbol_dictionary import enrich_with_symbols
    from structrag.retrieval.table_router import enrich_with_table_router

    retrieval_query = rewrite_query(question, use_llm=True)

    children = retrieve_child(retrieval_query, n_results=20, source_id=EC2_SOURCE_ID)
    relevant = [c for c in children if c.get("score", 999) < 0.8]
    chunks   = (
        resolve_parents(relevant, str(PARENT_CHUNKS_PATH))
        if relevant and PARENT_CHUNKS_PATH.exists()
        else []
    )

    hybrid = retrieve_hybrid(retrieval_query, n_results=20, source_id=EC2_SOURCE_ID)
    seen = {c["chunk_id"] for c in chunks}
    for h in hybrid:
        if h["chunk_id"] not in seen:
            chunks.append(h)
            seen.add(h["chunk_id"])

    chunks = rerank_with_fallback(question, chunks, top_n=8)
    chunks = enrich_with_table_router(question, chunks, EC2_SOURCE_ID)
    chunks = enrich_with_symbols(question, chunks, EC2_SOURCE_ID)

    formula = get_formula_for_query(question)
    if formula:
        m = re.search(r'\b(\d+\.\d+(?:\.\d+)*)\b', formula)
        fc = make_formula_chunk(formula, m.group(1) if m else "unknown", EC2_SOURCE_ID)
        if fc["chunk_id"] not in {c["chunk_id"] for c in chunks}:
            chunks.append(fc)

    return chunks


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG Génie Civil — Eurocode 2</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f1117; color: #e2e8f0;
      height: 100vh; display: flex; flex-direction: column;
    }
    header {
      padding: 16px 28px; border-bottom: 1px solid #1e2535;
      display: flex; align-items: center; gap: 12px;
    }
    header .logo {
      width: 32px; height: 32px;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      border-radius: 8px; display: flex; align-items: center;
      justify-content: center; font-size: 16px;
    }
    header h1 { font-size: 17px; font-weight: 600; color: #f1f5f9; }
    header span {
      font-size: 12px; color: #64748b;
      background: #1e2535; padding: 3px 10px; border-radius: 20px; margin-left: 4px;
    }
    #chat {
      flex: 1; overflow-y: auto; padding: 24px 0;
      display: flex; flex-direction: column; gap: 0;
    }
    #welcome {
      flex: 1; display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      gap: 12px; color: #475569; padding: 40px; text-align: center;
    }
    #welcome .icon { font-size: 48px; }
    #welcome h2 { font-size: 20px; color: #94a3b8; font-weight: 500; }
    #welcome p  { font-size: 14px; max-width: 420px; line-height: 1.6; }
    #welcome .examples { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 8px; }
    #welcome .ex {
      background: #1e2535; border: 1px solid #2d3748;
      padding: 8px 14px; border-radius: 8px; font-size: 13px;
      cursor: pointer; color: #94a3b8; transition: all .2s;
    }
    #welcome .ex:hover { background: #2d3748; color: #e2e8f0; }
    .msg-row { display: flex; padding: 10px 28px; gap: 14px; max-width: 860px; width: 100%; margin: 0 auto; }
    .msg-row.user { flex-direction: row-reverse; }
    .avatar {
      width: 32px; height: 32px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 14px; flex-shrink: 0; margin-top: 2px;
    }
    .user .avatar  { background: #3b82f6; }
    .model .avatar { background: linear-gradient(135deg, #3b82f6, #8b5cf6); }
    .bubble {
      max-width: 78%; padding: 12px 16px; border-radius: 14px;
      font-size: 14px; line-height: 1.7;
    }
    .user  .bubble { background: #1e3a5f; color: #e2e8f0; border-radius: 14px 4px 14px 14px; }
    .model .bubble { background: #1e2535; color: #e2e8f0; border-radius: 4px 14px 14px 14px; }
    .typing { display: flex; gap: 5px; align-items: center; padding: 4px 0; }
    .typing span {
      width: 7px; height: 7px; border-radius: 50%;
      background: #4a5568; animation: bounce .9s infinite;
    }
    .typing span:nth-child(2) { animation-delay: .15s; }
    .typing span:nth-child(3) { animation-delay: .30s; }
    @keyframes bounce { 0%,60%,100% { transform: translateY(0); } 30% { transform: translateY(-5px); } }
    .typing-label { font-size: 12px; color: #475569; margin-left: 8px; }
    #input-bar { padding: 16px 28px 20px; border-top: 1px solid #1e2535; }
    #input-wrap {
      max-width: 860px; margin: 0 auto;
      display: flex; gap: 10px; align-items: flex-end;
      background: #1e2535; border: 1px solid #2d3748;
      border-radius: 12px; padding: 10px 14px; transition: border-color .2s;
    }
    #input-wrap:focus-within { border-color: #3b82f6; }
    #question {
      flex: 1; background: transparent; border: none; outline: none;
      color: #e2e8f0; font-size: 14px; resize: none;
      max-height: 140px; min-height: 22px; line-height: 1.5; font-family: inherit;
    }
    #question::placeholder { color: #475569; }
    #send {
      background: #3b82f6; border: none; border-radius: 8px;
      width: 36px; height: 36px; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0; transition: background .2s;
    }
    #send:hover:not(:disabled) { background: #2563eb; }
    #send:disabled { background: #2d3748; cursor: not-allowed; }
    #send svg { width: 16px; height: 16px; fill: white; }
    #footer-note { text-align: center; font-size: 11px; color: #334155; margin-top: 8px; max-width: 860px; margin-left: auto; margin-right: auto; }
    .bubble { white-space: normal; }
    .bubble p  { margin-bottom: 8px; }
    .bubble p:last-child { margin-bottom: 0; }
    .bubble ul, .bubble ol { padding-left: 20px; margin-bottom: 8px; }
    .bubble li { margin-bottom: 3px; }
    .bubble strong { color: #93c5fd; }
    .bubble code {
      background: #0f2030; border: 1px solid #1e3a5f;
      padding: 1px 6px; border-radius: 4px; font-family: monospace; font-size: 13px; color: #7dd3fc;
    }
    .bubble pre {
      background: #0f2030; border: 1px solid #1e3a5f;
      padding: 10px 14px; border-radius: 8px; overflow-x: auto; margin: 8px 0;
    }
    .bubble pre code { background: none; border: none; padding: 0; }
    .bubble table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 13px; }
    .bubble th { background: #1e3a5f; padding: 6px 10px; text-align: left; color: #93c5fd; }
    .bubble td { padding: 5px 10px; border-top: 1px solid #1e2535; }
    #chat::-webkit-scrollbar { width: 5px; }
    #chat::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }
  </style>
</head>
<body>
<header>
  <div class="logo">⚙️</div>
  <h1>RAG Génie Civil</h1>
  <span>Eurocode 2 · NF EN 1992-1-1</span>
</header>
<div id="chat">
  <div id="welcome">
    <div class="icon">🏗️</div>
    <h2>Assistant Eurocode 2</h2>
    <p>Posez vos questions sur l'Eurocode 2 — enrobage, matériaux, dimensionnement, dispositions constructives.</p>
    <div class="examples">
      <div class="ex" onclick="ask(this)">Quelle est la formule de l'enrobage nominal ?</div>
      <div class="ex" onclick="ask(this)">Quelle est la valeur de γc en situation durable ?</div>
      <div class="ex" onclick="ask(this)">Quelles sont les classes d'exposition XC ?</div>
      <div class="ex" onclick="ask(this)">Quelle est la résistance fck pour C30/37 ?</div>
    </div>
  </div>
</div>
<div id="input-bar">
  <div id="input-wrap">
    <textarea id="question" rows="1" placeholder="Posez une question sur l'Eurocode 2…"></textarea>
    <button id="send" onclick="sendQuestion()">
      <svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>
    </button>
  </div>
  <div id="footer-note">Réponses basées uniquement sur NF EN 1992-1-1:2004 · Prototype de recherche</div>
</div>
<script>
  marked.setOptions({ breaks: true, gfm: true });
  const chat = document.getElementById('chat');
  const textarea = document.getElementById('question');
  const sendBtn  = document.getElementById('send');
  let timerInterval = null;

  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
  });
  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuestion(); }
  });

  function ask(el) { textarea.value = el.textContent; sendQuestion(); }

  function removeWelcome() {
    const w = document.getElementById('welcome');
    if (w) w.remove();
  }

  function addMessage(role, content) {
    removeWelcome();
    const row = document.createElement('div');
    row.className = `msg-row ${role}`;
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = role === 'model' ? marked.parse(content) : '';
    if (role === 'user') bubble.textContent = content;
    row.appendChild(avatar);
    row.appendChild(bubble);
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
  }

  function addTyping() {
    removeWelcome();
    const row = document.createElement('div');
    row.className = 'msg-row model';
    row.id = 'typing-row';
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '🤖';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = '<div style="display:flex;align-items:center"><div class="typing"><span></span><span></span><span></span></div><span class="typing-label" id="timer-label">Recherche en cours…</span></div>';
    row.appendChild(avatar);
    row.appendChild(bubble);
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
    let elapsed = 0;
    timerInterval = setInterval(() => {
      elapsed++;
      const label = document.getElementById('timer-label');
      if (label) label.textContent = elapsed < 5 ? 'Recherche en cours…' : elapsed < 15 ? `Récupération du contexte… (${elapsed}s)` : `Génération… (${elapsed}s)`;
    }, 1000);
  }

  function removeTyping() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
    const el = document.getElementById('typing-row');
    if (el) el.remove();
  }

  async function sendQuestion() {
    const q = textarea.value.trim();
    if (!q) return;
    addMessage('user', q);
    textarea.value = '';
    textarea.style.height = 'auto';
    sendBtn.disabled = true;
    addTyping();
    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      removeTyping();
      addMessage('model', data.error ? '⚠️ ' + data.error : data.answer);
    } catch (err) {
      removeTyping();
      addMessage('model', '⚠️ Erreur de connexion. Vérifiez que le serveur tourne.');
    } finally {
      sendBtn.disabled = false;
      textarea.focus();
    }
  }
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(_HTML)


@app.route("/ask", methods=["POST"])
def ask():
    data     = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question vide."}), 400

    try:
        _ensure_pipeline()

        from structrag.rag_pipeline.groq_generation import is_engineering_query
        if not is_engineering_query(question):
            return jsonify({
                "answer": (
                    "Cette question ne concerne pas l'Eurocode 2. "
                    "Je suis spécialisé dans les questions de génie civil "
                    "relatives à la norme NF EN 1992-1-1."
                ),
                "sources": [],
            })

        chunks = retrieve(question)
        if not chunks:
            return jsonify({
                "answer": "Aucun contexte pertinent trouvé. Veuillez reformuler votre question.",
                "sources": [],
            })

        from structrag.rag_pipeline.multi_model_generation import generate_multi
        result = generate_multi(
            question=question,
            chunks=chunks,
            model=MODEL_ID,
            provider=PROVIDER,
            temperature=0.01,
        )

        return jsonify({
            "answer":  result.answer,
            "sources": sorted(set(result.sources)) if result.sources else [],
        })

    except Exception as e:
        return jsonify({"error": f"Erreur interne: {str(e)[:200]}"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  RAG Génie Civil — Eurocode 2 Assistant")
    print(f"  Model  : {MODEL_ID} ({PROVIDER})")
    print("  Open   : http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
