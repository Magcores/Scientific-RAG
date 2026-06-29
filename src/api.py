"""FastAPI web server exposing the RAG pipeline over HTTP.

This is the production interface for the RAG chatbot.
Your website's chat widget sends questions here via HTTP POST,
and this server runs the RAG pipeline and sends the answer back.

Run locally:
    python src/api.py

Then open http://localhost:8080/docs to test it interactively.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "FastAPI is required. Install it with: pip install fastapi uvicorn"
    ) from exc

from rag import MAX_QUERIES_PER_MINUTE, MAX_QUERY_LENGTH, RAGPipeline, PERSONAS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Dr. Antúnez AI Assistant",
    description="Ask Mary or María anything about Dr. Antúnez's professional profile.",
)

# CORS — allows your website's JavaScript to call this API.
# In production, replace "*" with your actual website domain, e.g.:
# allow_origins=["https://www.martinantunez.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load the RAG pipeline once when the server starts (not on every request).
pipeline = RAGPipeline()

# Per-IP rate limiting — tracks timestamps of recent requests per visitor IP.
_ip_timestamps: dict[str, deque[float]] = defaultdict(deque)

# GCS bucket name — set this as an environment variable in Cloud Run.
# If empty, logging is silently skipped (safe for local development).
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")


# ---------------------------------------------------------------------------
# GCS logging
# ---------------------------------------------------------------------------

def log_to_gcs(query: str, answer: str, language: str) -> None:
    """
    Save one conversation entry to Google Cloud Storage.

    Each conversation is stored as a separate JSON file inside a
    folder named by date, e.g.:
        logs/2026-06-29/14-32-01_a3f8c2.json

    This makes it easy to browse by date in the Google Cloud Console
    and download any day's conversations.

    If GCS_BUCKET_NAME is not set (local development), this does nothing.
    If the upload fails for any reason, it logs the error but does NOT
    crash the API — a logging failure should never break the chat.
    """
    if not GCS_BUCKET_NAME:
        return

    try:
        from google.cloud import storage
    except ImportError:
        print("google-cloud-storage not installed — skipping GCS log.")
        return

    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "language": language,
            "query": query,
            "answer": answer,
        }

        # One folder per day, one file per conversation.
        date_folder = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        time_prefix = datetime.now(timezone.utc).strftime("%H-%M-%S")
        unique_id = uuid.uuid4().hex[:6]
        filename = f"logs/{date_folder}/{time_prefix}_{unique_id}.json"

        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(filename)
        blob.upload_from_string(
            json.dumps(entry, ensure_ascii=False, indent=2),
            content_type="application/json",
        )
    except Exception as exc:
        # Never crash the API because of a logging failure.
        print(f"GCS logging failed: {exc}")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def check_rate_limit(ip: str) -> None:
    """Raise HTTP 429 if this IP has exceeded the per-minute request limit."""
    now = time.monotonic()
    timestamps = _ip_timestamps[ip]
    while timestamps and now - timestamps[0] > 60:
        timestamps.popleft()
    if len(timestamps) >= MAX_QUERIES_PER_MINUTE:
        wait = int(60 - (now - timestamps[0])) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait {wait} seconds before asking again.",
        )
    timestamps.append(now)


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------

class HistoryTurn(BaseModel):
    """One previous question-answer pair, sent by the client to maintain context."""
    query: str
    answer: str


class Question(BaseModel):
    """
    What the website sends to the API.

    - query:    The visitor's question.
    - language: "en" for Mary (English) or "es" for María (Spanish).
    - history:  The conversation so far (last few turns), so follow-up
                questions work correctly. The client (website) is responsible
                for maintaining and sending this list.
    """
    query: str
    language: str = "en"
    history: list[HistoryTurn] = []


class Answer(BaseModel):
    """What the API sends back to the website."""
    answer: str
    query: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    """Health check — Cloud Run uses this to know the server is alive."""
    return {"status": "ok", "message": "Dr. Antúnez AI Assistant is running."}


@app.post("/ask", response_model=Answer)
def ask(question: Question, request: Request):
    """
    Main endpoint. The website POSTs a question here and gets an answer back.

    Example request body:
        {
            "query": "What cloud platforms has Dr. Antúnez worked with?",
            "language": "es",
            "history": [
                {"query": "previous question", "answer": "previous answer"}
            ]
        }
    """
    check_rate_limit(request.client.host if request.client else "unknown")

    if len(question.query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question is too long ({len(question.query)} characters). Maximum is {MAX_QUERY_LENGTH}.",
        )

    lang = question.language if question.language in PERSONAS else "en"
    persona = PERSONAS[lang]

    history = [{"query": t.query, "answer": t.answer} for t in question.history]

    answer = pipeline.answer(
        query=question.query,
        system_instructions=persona["system_instructions"],
        no_results_message=persona["no_results"],
        history=history or None,
    )

    log_to_gcs(question.query, answer, lang)

    return Answer(answer=answer, query=question.query)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("Install uvicorn with: pip install uvicorn") from exc

    # PORT is set automatically by Cloud Run. Locally it defaults to 8080.
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
