# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# A Dockerfile is a recipe. Docker reads it top to bottom and builds a
# "container image" — a self-contained package with Python, your code,
# your libraries, and your ChromaDB database all bundled together.
#
# Cloud Run will run that container image in Google's infrastructure.
# ─────────────────────────────────────────────────────────────────────────────


# STEP 1 — Choose a base image.
# Think of this as the starting point: a clean Linux machine with Python 3.11
# already installed. "slim" means it's a minimal version (smaller = faster).
FROM python:3.11-slim


# STEP 2 — Set the working directory inside the container.
# All following commands run from this folder.
WORKDIR /app


# STEP 3 — Copy requirements.txt first and install dependencies.
# We do this BEFORE copying the rest of the code because Docker caches each
# step. If you only change your code (not requirements.txt), Docker skips
# this step on the next build and reuses the cached libraries — much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# STEP 4 — Copy your source code into the container.
COPY src/ ./src/


# STEP 5 — Copy the ChromaDB index (the vector database you built with ingest.py).
# IMPORTANT: You must run ingest.py locally BEFORE building this Docker image,
# so that data/index/chroma_db/ exists on your machine to be copied in.
COPY data/index/chroma_db/ ./data/index/chroma_db/


# STEP 6 — Tell Docker which port the server listens on.
# Cloud Run always sends traffic to port 8080. This line is documentation
# more than a rule — the actual port is set in api.py via the PORT env var.
EXPOSE 8080


# STEP 7 — The command that runs when the container starts.
# This starts your FastAPI web server.
CMD ["python", "src/api.py"]
