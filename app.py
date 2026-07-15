"""
CRSIJ Chatbot Backend
Lightweight Flask replacement for the n8n workflow.
Flow: receive message -> retrieve relevant chunks from Pinecone -> call Gemini -> save to Supabase -> respond
"""

import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import Json

app = Flask(__name__)
CORS(app)  # allow requests from your chatbot's frontend (any origin, for now)

# ---- CONFIG (set these as environment variables on your host) ----
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.environ.get("PINECONE_INDEX_HOST")  # e.g. https://medical-chatbot-xxxx.svc.xxxx.pinecone.io

HF_TOKEN = os.environ.get("HF_TOKEN")  # Hugging Face token for embeddings
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATABASE_URL = os.environ.get("DATABASE_URL")  # Supabase Postgres connection string

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
        vector_len = len(embedding[0])
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


def get_chat_history(session_id, limit=10):
    """Fetch recent chat history for this session from Supabase.
    Returns an empty list (instead of raising) if DATABASE_URL is missing or the
    connection fails, so the chatbot still works without memory rather than erroring out."""
    if not DATABASE_URL:
        return []
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT message FROM n8n_chat_histories
                    WHERE session_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (session_id, limit),
                )
                rows = cur.fetchall()
            return [row[0] for row in reversed(rows)]
        finally:
            conn.close()
    except Exception:
        app.logger.exception("get_chat_history failed - continuing without memory")
        return []


def save_message(session_id, message_type, content):
    """Save a message (human or ai) to Supabase, matching the existing table structure.
    Silently no-ops if DATABASE_URL is missing or the connection fails, so a DB issue
    never breaks the actual chat response."""
    if not DATABASE_URL:
        return
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO n8n_chat_histories (session_id, message)
                    VALUES (%s, %s)
                    """,
                    (
                        session_id,
                        Json(
                            {
                                "type": message_type,
                                "content": content,
                                "additional_kwargs": {},
                                "response_metadata": {},
                            }
                        ),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        app.logger.exception("save_message failed - continuing without saving")


def call_gemini(user_message, context_chunks, history):
    """Call Gemini API with system prompt, retrieved context, and chat history."""
    context_text = "\n\n".join(context_chunks) if context_chunks else ""

    contents = []
    for msg in history:
        role = "user" if msg.get("type") == "human" else "model"
        contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

    final_user_text = user_message
    if context_text:
        final_user_text = f"Context:\n{context_text}\n\nQuestion: {user_message}"

    contents.append({"role": "user", "parts": [{"text": final_user_text}]})

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
    }

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


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

        # 4. Call Gemini
        reply = call_gemini(user_message, context_chunks, history)

        # 5. Save both messages to Supabase
        save_message(session_id, "human", user_message)
        save_message(session_id, "ai", reply)

        return jsonify({"output": reply})

    except Exception as e:
        app.logger.exception("Error in /webhook/chat")
        return jsonify({"output": "Sorry, something went wrong on my end. Please try again in a moment."}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Only used for local testing (python app.py).
    # Vercel imports the 'app' object directly and does not call this.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
