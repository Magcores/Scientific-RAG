"""FastAPI web server exposing the RAG pipeline over HTTP.

This is an optional interface — the RAG logic lives entirely in rag.py.
This file just adds an HTTP layer on top so the pipeline can be called
from a website, a mobile app, or any other HTTP client.

Setup:
    pip install fastapi uvicorn

Run:
    python src\api.py

Then send questions to:
    POST http://localhost:8000/ask
    Body: {"query": "What is theta activity?"}

Or open http://localhost:8000/docs for an interactive interface.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict, deque
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
        "FastAPI is required to run the web server. "
        "Install it with: pip install fastapi uvicorn"
    ) from exc

from rag import MAX_QUERIES_PER_MINUTE, MAX_QUERY_LENGTH, RAGPipeline

app = FastAPI(
    title="RAG API",
    description="Ask questions over indexed local documents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()

# Per-IP rate limiting — same window logic as the interactive CLI
_ip_timestamps: dict[str, deque[float]] = defaultdict(deque)


def check_rate_limit(ip: str) -> None:
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


class Question(BaseModel):
    query: str


class Answer(BaseModel):
    answer: str
    query: str


@app.get("/")
def health():
    return {"status": "ok", "message": "RAG API is running. POST /ask to query."}


@app.post("/ask", response_model=Answer)
def ask(question: Question, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    if len(question.query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Query is too long ({len(question.query)} characters). Maximum is {MAX_QUERY_LENGTH}.",
        )

    answer = pipeline.answer(question.query)
    return Answer(answer=answer, query=question.query)


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required to run the web server. "
            "Install it with: pip install uvicorn"
        ) from exc

    uvicorn.run(app, host="127.0.0.1", port=8000)
