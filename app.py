"""
CRSIJ Chatbot Backend
Lightweight Flask backend for the chatbot.
Flow: receive message -> retrieve relevant chunks from Pinecone -> call OpenRouter -> respond
Also stores/retrieves chat history per session in Neon Postgres.
"""

import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2

app = Flask(__name__)

# Only allow requests from your actual site - replace this with your real domain(s).
# Include both with and without "www." if your site uses either.
ALLOWED_ORIGINS = [
    "extraordinary-puffpuff-751ad7.netlify.app",
    "https://digitaldb.in",
    "http://localhost:8000"
]
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# ---- CONFIG (set these as environment variables on your host) ----
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Any OpenRouter model slug works here. Default is the auto-router, which picks a
# strong available model for you (including free ones) so you're not locked into
# one specific model. Override via env var if you want to pin a specific model,
# e.g. OPENROUTER_MODEL=anthropic/claude-3.5-sonnet or any other slug.
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_TEMPERATURE = float(os.environ.get("OPENROUTER_TEMPERATURE", "0.2"))
OPENROUTER_MAX_TOKENS = int(os.environ.get("OPENROUTER_MAX_TOKENS", "150"))

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.environ.get("PINECONE_INDEX_HOST")  # e.g. https://medical-chatbot-xxxx.svc.xxxx.pinecone.io
PINECONE_TOP_K = int(os.environ.get("PINECONE_TOP_K", "4"))

HF_TOKEN = os.environ.get("HF_TOKEN")  # Hugging Face token for embeddings
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATABASE_URL = os.environ.get("NEON_DATABASE_URL")  # Neon Postgres connection string
CHAT_HISTORY_LIMIT = int(os.environ.get("CHAT_HISTORY_LIMIT", "6"))  # turns, not messages

# NOTE: written as clean, non-overlapping sentences on purpose - the previous
# version had two instructions spliced into each other mid-sentence, which
# produced confusing/self-contradictory guidance for the model.
SYSTEM_PROMPT = (
    "LINKS: Whenever the search results contain a URL relevant to the user's question, you MUST format it as a Markdown link: [descriptive text](URL) — for example [Submit your paper here](https://cosmorsij.com/submit-paper). NEVER write a bare URL by itself (e.g. "https://cosmorsij.com/submit-paper" alone is WRONG). Only use a URL that appears exactly in the search results — never invent one."
    "Do not mention your internal tools or data sources."
)


def get_embedding(text):
    """Get embedding vector from HuggingFace Inference API (same model used in Pinecone index)."""
    response = requests.post(
        f"https://router.huggingface.co/hf-inference/models/{HF_EMBEDDING_MODEL}/pipeline/feature-extraction",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": text},
        timeout=30,
    )
    response.raise_for_status()
    embedding = response.json()
    # Some HF endpoints return nested lists (token-level); average-pool if needed
    if isinstance(embedding[0], list):
        avg = [sum(col) / len(embedding) for col in zip(*embedding)]
        return avg
    return embedding


def query_pinecone(vector, top_k=PINECONE_TOP_K):
    """Query Pinecone for the most relevant chunks.
    Returns an empty list (instead of raising) on any failure, so the chatbot still
    answers gracefully rather than erroring out."""
    try:
        response = requests.post(
            f"{PINECONE_INDEX_HOST}/query",
            headers={"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"},
            json={"vector": vector, "topK": top_k, "includeMetadata": True},
            timeout=30,
        )
        response.raise_for_status()
        matches = response.json().get("matches", [])
        return [m.get("metadata", {}).get("text", "") for m in matches if m.get("metadata")]
    except Exception:
        app.logger.exception("query_pinecone failed - continuing without retrieved context")
        return []


def query_pinecone_raw(vector, top_k=PINECONE_TOP_K):
    """Same as query_pinecone but re-raises errors and returns full match objects
    (score + metadata), for debugging only - not used in the main chat flow."""
    response = requests.post(
        f"{PINECONE_INDEX_HOST}/query",
        headers={"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"},
        json={"vector": vector, "topK": top_k, "includeMetadata": True},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("matches", [])


def get_chat_history(session_id, limit=CHAT_HISTORY_LIMIT):
    """Fetch recent chat history for this session from Neon."""
    if not DATABASE_URL:
        return []
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_input, bot_response FROM chat_histories
                WHERE session_id = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()
        return list(reversed(rows))
    except Exception:
        app.logger.exception("get_chat_history failed - continuing without history")
        return []
    finally:
        conn.close()


def save_exchange(session_id, user_input, bot_response):
    """Save one full exchange (user input + bot response) as a single row."""
    if not DATABASE_URL:
        return
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_histories (session_id, user_input, bot_response)
                VALUES (%s, %s, %s)
                """,
                (session_id, user_input, bot_response),
            )
        conn.commit()
    except Exception:
        app.logger.exception("save_exchange failed - history not persisted for this turn")
    finally:
        conn.close()


def call_openrouter(user_message, context_chunks, history):
    """Call OpenRouter's chat completions API (OpenAI-compatible format) with
    system prompt, retrieved context, and chat history."""
    context_text = "\n\n".join(context_chunks) if context_chunks else ""

    final_user_text = user_message
    if context_text:
        final_user_text = f"Context:\n{context_text}\n\nQuestion: {user_message}"
    else:
        final_user_text = (
            f"Context: (no relevant context was found)\n\nQuestion: {user_message}"
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_input, bot_response in history:
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": bot_response})
    messages.append({"role": "user", "content": final_user_text})

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": OPENROUTER_TEMPERATURE,
        "max_tokens": OPENROUTER_MAX_TOKENS,
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


@app.route("/webhook/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = data.get("chatInput", "")
        session_id = data.get("sessionId", "default-session")

        if not user_message:
            return jsonify({"output": "I didn't receive a message. Could you try again?"}), 400

        # 1. Embed the user's question (skip retrieval entirely if this fails)
        context_chunks = []
        try:
            vector = get_embedding(user_message)
            # 2. Retrieve relevant chunks from Pinecone
            context_chunks = query_pinecone(vector)
        except Exception:
            app.logger.exception("Embedding step failed - continuing without retrieved context")

        # 3. Get recent chat history
        history = get_chat_history(session_id)

        # 4. Call OpenRouter
        reply = call_openrouter(user_message, context_chunks, history)

        # 5. Save this exchange
        save_exchange(session_id, user_message, reply)

        return jsonify({"output": reply})

    except Exception:
        app.logger.exception("Error in /webhook/chat")
        return jsonify({"output": "Sorry, something went wrong on my end. Please try again in a moment."}), 500


@app.route("/debug/retrieve", methods=["GET"])
def debug_retrieve():
    """
    Debug-only route: bypasses OpenRouter entirely and shows you exactly what
    Pinecone returns for a given query, plus each match's similarity score.
    Usage: GET /debug/retrieve?q=publication%20charges

    Remove or protect this route before going fully to production, since it
    exposes raw chunk text from your index.
    """
    query_text = request.args.get("q", "")
    if not query_text:
        return jsonify({"error": "pass a query string, e.g. /debug/retrieve?q=publication charges"}), 400

    debug_info = {"query": query_text}

    # Step 1: embedding
    try:
        vector = get_embedding(query_text)
        debug_info["embedding_ok"] = True
        debug_info["embedding_length"] = len(vector)
    except Exception as e:
        debug_info["embedding_ok"] = False
        debug_info["embedding_error"] = str(e)
        return jsonify(debug_info), 200

    # Step 2: pinecone query (raw, errors not swallowed)
    try:
        matches = query_pinecone_raw(vector, top_k=PINECONE_TOP_K)
        debug_info["pinecone_ok"] = True
        debug_info["match_count"] = len(matches)
        debug_info["matches"] = [
            {
                "score": m.get("score"),
                "id": m.get("id"),
                "text_preview": (m.get("metadata", {}).get("text", "") or "")[:300],
            }
            for m in matches
        ]
    except Exception as e:
        debug_info["pinecone_ok"] = False
        debug_info["pinecone_error"] = str(e)

    return jsonify(debug_info), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Only used for local testing (python app.py).
    # Vercel imports the 'app' object directly and does not call this.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
