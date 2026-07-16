"""
CRSIJ Chatbot Backend
Lightweight Flask replacement for the n8n workflow.
Flow: receive message -> retrieve relevant chunks from Pinecone -> call OpenRouter -> respond
No database - each request is handled statelessly (no chat history stored).
/sync endpoint replaces the old n8n workflow: deletes all Pinecone vectors, then
re-embeds and re-uploads fresh text sent from the Google Apps Script.
"""

import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# ---- ACCESS CONTROL: only allow your website to call this API ----
# Set ALLOWED_ORIGIN to your site's exact origin, e.g. "https://www.yourjournalsite.com"
# (no trailing slash, must include https://)
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "")

CORS(app, origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN else [], supports_credentials=False)

# ---- CONFIG (set these as environment variables on your host / Vercel project settings) ----
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Any OpenRouter model slug works here. Use a ":free" model if you want $0 cost,
# e.g. "meta-llama/llama-3.1-8b-instruct:free" or "openrouter/free" (auto-router).
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")

# Optional but recommended by OpenRouter for attribution/rankings - not required to work.
SITE_URL = os.environ.get("SITE_URL", "")
SITE_NAME = os.environ.get("SITE_NAME", "CRSIJ Chatbot")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.environ.get("PINECONE_INDEX_HOST")  # e.g. https://medical-chatbot-xxxx.svc.xxxx.pinecone.io

HF_TOKEN = os.environ.get("HF_TOKEN")  # Hugging Face token for embeddings
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Secret shared only between your Google Apps Script and this backend, used to
# authorize the /sync endpoint (Apps Script calls don't send Origin/Referer
# headers like a browser does, so the Origin check used for /webhook/chat
# doesn't apply here - this endpoint needs its own check).
SYNC_SECRET = os.environ.get("SYNC_SECRET", "")

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


def call_openrouter(user_message, context_chunks):
    """Call OpenRouter (OpenAI-compatible chat completions API) with system prompt
    and retrieved context. No persistent chat history - each request is stateless."""
    context_text = "\n\n".join(context_chunks) if context_chunks else ""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    final_user_text = user_message
    if context_text:
        final_user_text = f"Context:\n{context_text}\n\nQuestion: {user_message}"

    messages.append({"role": "user", "content": final_user_text})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if SITE_URL:
        headers["HTTP-Referer"] = SITE_URL
    if SITE_NAME:
        headers["X-Title"] = SITE_NAME

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
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
                # Paragraph itself is too long - slice it with overlap
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
    as metadata (so query_pinecone can return the text directly, same as before)."""
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



def verify_origin():
    """Reject any request that doesn't claim to come from ALLOWED_ORIGIN.
    CORS alone only stops browsers from *reading* the response - it does not stop
    curl/Postman/other servers from calling this endpoint directly. Checking the
    Origin/Referer header blocks casual direct calls too. Note: a determined caller
    can still fake these headers, so for real protection also rotate your
    OpenRouter key if you ever see abuse, and consider adding a shared-secret
    header (e.g. a value you also set in your frontend's fetch call) for stronger
    protection than Origin/Referer checks alone."""
    if not ALLOWED_ORIGIN:
        # TEMPORARY: allow all origins if ALLOWED_ORIGIN isn't set correctly yet.
        # This restores basic functionality immediately - re-enable the strict
        # check once ALLOWED_ORIGIN is confirmed to exactly match your site.
        return True

    origin = request.headers.get("Origin", "")
    referer = request.headers.get("Referer", "")

    if origin == ALLOWED_ORIGIN:
        return True
    if referer.startswith(ALLOWED_ORIGIN):
        return True
    return False


@app.route("/webhook/chat", methods=["POST"])
def chat():
    if not verify_origin():
        return jsonify({"output": "Forbidden"}), 403

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

        # 3. Call OpenRouter (stateless - no chat history stored)
        reply = call_openrouter(user_message, context_chunks)

        return jsonify({"output": reply})

    except Exception as e:
        app.logger.exception("Error in /webhook/chat")
        return jsonify({"output": "Sorry, something went wrong on my end. Please try again in a moment."}), 500


@app.route("/sync", methods=["POST"])
def sync():
    # Auth check: require the shared secret instead of Origin/Referer, since
    # Apps Script's UrlFetchApp doesn't send those headers.
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
