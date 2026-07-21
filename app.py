"""
CRSIJ Chatbot Backend
Lightweight Flask backend for the chatbot.
Flow: receive message -> retrieve relevant chunks from Pinecone -> call OpenRouter -> respond
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
# Any OpenRouter model slug works here. Use a ":free" model if you want $0 cost,
# e.g. "meta-llama/llama-3.1-8b-instruct:free" or "openrouter/free" (auto-router).
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_TEMPERATURE = float(os.environ.get("OPENROUTER_TEMPERATURE", "0.7"))
OPENROUTER_MAX_TOKENS = int(os.environ.get("OPENROUTER_MAX_TOKENS", "500"))

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.environ.get("PINECONE_INDEX_HOST")  # e.g. https://medical-chatbot-xxxx.svc.xxxx.pinecone.io

HF_TOKEN = os.environ.get("HF_TOKEN")  # Hugging Face token for embeddings
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATABASE_URL = os.environ.get("NEON_DATABASE_URL")  # Neon Postgres connection string

SYSTEM_PROMPT = (
    "You are the CRSI Journal assistant. You must answer strictly using the CONTEXT "
    "provided below each question - that context comes from the official CRSI Journal "
    "documents. Do not use outside/general knowledge about academic publishing in general; "
    "only use what is explicitly stated in the context. "
    "If the context includes a URL that is directly relevant to the question (e.g. a "
    "submission page, guidelines page, or tracking portal), include that exact URL in your "
    "If the context does not contain the answer, say clearly that you don't have that "
    "specific information and suggest the user contact the journal directly, rather than "
    "guessing or giving generic advice. "
    "answer. Do not invent or add a link that isn't present in the context. "
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


def query_pinecone(vector, top_k=8):
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
    finally:
        conn.close()


def call_openrouter(user_message, context_chunks, history):
    """Call OpenRouter's chat completions API (OpenAI-compatible format) with
    system prompt, retrieved context, and chat history."""
    context_text = "\n\n".join(context_chunks) if context_chunks else ""

    final_user_text = user_message
    if context_text:
        final_user_text = f"Context:\n{context_text}\n\nQuestion: {user_message}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_input, bot_response in history:
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": bot_response})
    messages.append({"role": "user", "content": final_user_text})

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 280,
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
