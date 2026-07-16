"""
CRSIJ Chatbot Backend
Lightweight Flask backend for the chatbot.
Flow: receive message -> retrieve relevant chunks from Pinecone -> call OpenRouter -> respond
(No chat memory/database - each message is answered independently using Pinecone context.)
"""

import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Only allow requests from your actual site - replace this with your real domain(s).
# Include both with and without "www." if your site uses either.
ALLOWED_ORIGINS = [
    "extraordinary-puffpuff-751ad7.netlify.app",
    "https://www.YOUR-SITE-DOMAIN-HERE.com",
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


@app.route("/webhook/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = data.get("chatInput", "")

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
