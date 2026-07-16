"""
CRSIJ Chatbot Backend
Lightweight Flask backend for the chatbot.
Flow: receive message -> retrieve relevant chunks from Pinecone -> call OpenRouter -> log to Neon -> respond
Chat messages (both user and bot) are logged to Neon Postgres with an India-time timestamp.
/sync endpoint: deletes all Pinecone vectors, then re-embeds and re-uploads fresh text
sent from the Google Apps Script.
"""

import os
import requests
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2

app = Flask(__name__)

# Only allow requests from your actual site - replace this with your real domain(s).
# Include both with and without "www." if your site uses either.
ALLOWED_ORIGINS = [
    "https://extraordinary-puffpuff-751ad7.netlify.app",
    "https://www.YOUR-SITE-DOMAIN-HERE.com",
    "http://localhost:8000"
]
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# ---- CONFIG (set these as environment variables on your host) ----
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Any OpenRouter model slug works here. Use a ":free" model if you want $0 cost,
# e.g. "meta-llama/llama-3.1-8b-instruct:free" or "openrouter/free" (auto-router).
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.environ.get("PINECONE_INDEX_HOST")  # e.g. https://medical-chatbot-xxxx.svc.xxxx.pinecone.io

HF_TOKEN = os.environ.get("HF_TOKEN")  # Hugging Face token for embeddings
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Secret shared only between your Google Apps Script and this backend, used to
# authorize the /sync endpoint.
SYNC_SECRET = os.environ.get("SYNC_SECRET", "")

# Neon Postgres connection string, e.g.
# postgresql://user:password@ep-xxxx.neon.tech/dbname?sslmode=require
NEON_DATABASE_URL = os.environ.get("NEON_DATABASE_URL")

# India Standard Time is a fixed UTC+5:30 offset with no daylight saving,
# so a simple fixed-offset timezone is accurate year-round.
IST = timezone(timedelta(hours=5, minutes=30))

SYSTEM_PROMPT = (
    "You are the CRSI Journal assistant. Answer questions about submitting a paper, "
    "tracking submissions, and publication charges. Use the provided context if relevant. "
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


def query_pinecone(vector, top_k=4):
    """Query Pinecone for the most relevant chunks.
    Returns an empty list (instead of raising) on any failure, so the chatbot still
    answers using general knowledge rather than erroring out."""
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


def call_openrouter(user_message, context_chunks):
    """Call OpenRouter's chat completions API (OpenAI-compatible format) with
    system prompt and retrieved context."""
    context_text = "\n\n".join(context_chunks) if context_chunks else ""

    final_user_text = user_message
    if context_text:
        final_user_text = f"Context:\n{context_text}\n\nQuestion: {user_message}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": final_user_text},
    ]

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
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


def chunk_text(text, max_chars=1500, overlap=200):
    """Split a long document into overlapping chunks so each one embeds cleanly
    and retrieval can find the right paragraph. Splits on paragraph breaks first,
    falling back to raw character slicing if a single paragraph is too long."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                start = 0
                while start < len(para):
                    chunks.append(para[start:start + max_chars])
                    start += max_chars - overlap
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


def delete_all_vectors(namespace=""):
    """Delete every vector in the given Pinecone namespace before re-uploading
    fresh data, so old/removed content never lingers in the index."""
    response = requests.post(
        f"{PINECONE_INDEX_HOST}/vectors/delete",
        headers={"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"},
        json={"deleteAll": True, "namespace": namespace},
        timeout=30,
    )
    response.raise_for_status()


def upsert_chunks(chunks, namespace=""):
    """Embed each chunk and upsert it into Pinecone with the chunk text stored
    as metadata (so query_pinecone can return the text directly)."""
    batch_size = 50
    vectors = []

    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        vectors.append({
            "id": f"chunk-{i}",
            "values": vector,
            "metadata": {"text": chunk},
        })

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        response = requests.post(
            f"{PINECONE_INDEX_HOST}/vectors/upsert",
            headers={"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"},
            json={"vectors": batch, "namespace": namespace},
            timeout=60,
        )
        response.raise_for_status()

    return len(vectors)


def log_message(session_id, message_type, content):
    """Save a chat message (user or bot) to Neon Postgres with an India-time
    timestamp. Silently no-ops if NEON_DATABASE_URL is missing or the
    connection fails, so a DB issue never breaks the actual chat response."""
    if not NEON_DATABASE_URL:
        return
    try:
        conn = psycopg2.connect(NEON_DATABASE_URL, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_logs (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT,
                        message_type TEXT,
                        content TEXT,
                        created_at TIMESTAMPTZ
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO chat_logs (session_id, message_type, content, created_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (session_id, message_type, content, datetime.now(IST)),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        app.logger.exception("log_message failed - continuing without logging")


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

        # 3. Call OpenRouter
        reply = call_openrouter(user_message, context_chunks)

        # 4. Log both messages to Neon (no-ops if NEON_DATABASE_URL isn't set)
        log_message(session_id, "human", user_message)
        log_message(session_id, "ai", reply)

        return jsonify({"output": reply})

    except Exception as e:
        app.logger.exception("Error in /webhook/chat")
        return jsonify({"output": "Sorry, something went wrong on my end. Please try again in a moment."}), 500


@app.route("/sync", methods=["POST"])
def sync():
    # Auth check: require the shared secret since Apps Script's UrlFetchApp
    # doesn't send Origin/Referer headers like a browser does.
    data = request.get_json(force=True, silent=True) or {}
    provided_secret = request.headers.get("X-Sync-Secret") or data.get("secret", "")

    if not SYNC_SECRET or provided_secret != SYNC_SECRET:
        return jsonify({"status": "error", "message": "Forbidden"}), 403

    document_text = data.get("text", "")
    if not document_text.strip():
        return jsonify({"status": "error", "message": "No document text provided"}), 400

    try:
        delete_all_vectors()
    except Exception:
        app.logger.exception("delete_all_vectors failed")
        return jsonify({"status": "error", "message": "Failed to delete old Pinecone data"}), 500

    try:
        chunks = chunk_text(document_text)
        count = upsert_chunks(chunks)
    except Exception:
        app.logger.exception("upsert_chunks failed")
        return jsonify({"status": "error", "message": "Deleted old data, but failed to upload new data"}), 500

    return jsonify({"status": "ok", "message": f"Synced {count} chunks to Pinecone"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Only used for local testing (python app.py).
    # Vercel imports the 'app' object directly and does not call this.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
