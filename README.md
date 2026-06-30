# RAG Chatbot — Production-Ready AI Assistant

A production-grade Retrieval-Augmented Generation (RAG) system that powers a personal website chatbot. Visitors ask questions in plain English and get answers grounded strictly in a curated document knowledge base, served via a FastAPI backend deployed on Google Cloud Run.

---

## Tech Stack

**AI / ML**
- [OpenAI API](https://platform.openai.com) — `text-embedding-3-small` for semantic embeddings, `gpt-4o-mini` for answer generation
- [ChromaDB](https://www.trychroma.com) — local vector database for semantic similarity search
- Custom RAG pipeline with diversity-aware retrieval (per-source chunk capping, cosine similarity scoring)

**Backend**
- [FastAPI](https://fastapi.tiangolo.com) — REST API with automatic `/docs` UI
- [Uvicorn](https://www.uvicorn.org) — ASGI server
- [Pydantic](https://docs.pydantic.dev) — request/response validation
- Per-IP rate limiting, input length validation, multi-language persona system (English / Spanish)

**Document Processing**
- [pypdf](https://pypdf.readthedocs.io) — PDF text extraction
- [ftfy](https://ftfy.readthedocs.io) — encoding artifact repair
- Custom recursive chunking with overlap, header/footer deduplication, reference section stripping

**Infrastructure & Deployment**
- [Docker](https://www.docker.com) / [Docker Desktop](https://www.docker.com/products/docker-desktop) — containerized application, local build and test
- [Google Cloud Run](https://cloud.google.com/run) — serverless container deployment, autoscaling, zero cold-start config
- [Google Artifact Registry](https://cloud.google.com/artifact-registry) — private Docker image registry
- [Google Cloud Storage](https://cloud.google.com/storage) — conversation log storage (one JSON file per request, organized by date, auto-deleted after 90 days)
- [Google Secret Manager](https://cloud.google.com/secret-manager) — secure API key injection at runtime
- [Google Cloud SDK (`gcloud` CLI)](https://cloud.google.com/sdk) — provisioning, IAM permissions, deployments from terminal

**Language & Runtime**
- Python 3.11
- [python-dotenv](https://github.com/theskumar/python-dotenv) — local environment variable management

**Testing**
- [pytest](https://pytest.org) — unit tests for chunking, PDF cleaning, metadata extraction (offline, no API calls)

**Frontend**
- Bilingual UI — English (Mary) and Spanish (María) personas
- Conversation history maintained client-side across turns

---

## What it looks like

Once set up, you run it interactively and it feels like this:

```
> What is the role of theta activity in cognitive functioning?

Theta activity (4–8 Hz) is strongly associated with cognitive functioning,
particularly working memory and attention. Studies show that theta power
increases during memory encoding tasks and is linked to executive function
in both typical and atypical populations.

Sources:
- Tan et al. 2024, page 2 (score: 0.87)
- Antúnez et al. 2021, page 5 (score: 0.81)

[1/20 questions used this session]

> quit
Goodbye.
```

---

## File structure

**Input — what you put in:**
```
data/documents/
├── example.txt          ← placeholder, replace with your own files
├── your_paper.pdf       ← drop any .pdf or .txt files here before ingesting
└── your_notes.txt
```

**Output — what the system creates after running `ingest.py`:**
```
data/index/
├── chroma_db/                  ← the vector database (created by ingest.py)
│   ├── chroma.sqlite3          ← chunk text and metadata
│   └── [uuid]/
│       └── *.bin               ← embedding vectors (binary, not human-readable)
└── chunks_preview.json         ← only created by --dry-run, for inspection only
```

`data/documents/` is tracked by git (only `example.txt` is included). `data/index/` is excluded from git entirely — it is created automatically the first time you run `ingest.py`.

---

## What you need to get started

- Python 3.x
- An OpenAI API key
- A `.env` file in the project root (copy `.env.example` and fill in your key):
  ```
  OPENAI_API_KEY=sk-...
  ```
- Dependencies installed:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

Put your `.pdf` or `.txt` files in the `data/documents/` folder before running anything.

---

## 1. `src/ingest.py` — reads your documents and builds the search index

This is the first script you run. It goes through every file in `data/documents/`, cleans up the raw text (fixing encoding issues, removing repeated headers and footers, stripping reference sections), and splits everything into overlapping chunks small enough to be individually meaningful.

Once it has the chunks, it sends them to OpenAI to create semantic embeddings — along with the chunk text and metadata like source file, page number, title, and year — and stores everything in a local **Chroma** database under `data/index/`. Chroma is a vector database built for this kind of similarity search.

You only need to run this once, or again whenever you add new documents.

Key tools used: `pypdf` to extract text from PDFs, `ftfy` to repair encoding artifacts, `openai` for embeddings, `chromadb` to store and index everything locally.

```powershell
# Check how your documents will be chunked before spending on API calls (free):
python src\ingest.py --dry-run

# Build the index:
python src\ingest.py

# Add new documents without rebuilding from scratch:
python src\ingest.py --append

# Keep reference sections (they are stripped by default):
python src\ingest.py --keep-references
```

---

## 2. `src/rag.py` — asks questions and generates answers

This is the script you use day to day. You give it a question, it embeds that question the same way the chunks were embedded, searches the Chroma database for the most semantically similar chunks, and passes those chunks to an OpenAI chat model along with your question. The result is a short, grounded answer plus citations telling you which document and page it came from.

It has two modes: pass a question directly and get one answer, or run it without a question to enter an interactive session where you keep asking until you type `quit`. The interactive mode loads the pipeline once and keeps it in memory, so follow-up questions are faster.

A few limits are built in to avoid runaway costs: questions are capped at 500 characters, answers at 900 tokens, and the interactive session allows a maximum of 10 questions per minute and 20 questions total before asking you to restart.

Key tools used: `openai` to embed the query and generate the answer, `chromadb` to search the index.

```powershell
# Ask a single question:
python src\rag.py "What is theta activity related to?"

# Interactive mode (keep asking questions):
python src\rag.py

# See the retrieved chunks before the answer:
python src\rag.py "your question" --show-context

# Retrieve more chunks (default is 10):
python src\rag.py "your question" --top-k 15
```

---

## 3. `src/faithfulness.py` — measures how much the answers hallucinate

This is an optional evaluation script. It runs a set of predefined questions through the full pipeline and then checks whether each claim in the generated answers is actually supported by the retrieved chunks. A claim that cannot be traced back to the sources is a hallucination.

It works by making two types of OpenAI calls per question: one to break the answer into individual factual claims, and one per claim to ask whether the retrieved context supports it. The result is a score between 0 and 1 per question, plus a per-claim breakdown showing exactly which parts of the answer were grounded and which were not.

You can edit the `SAMPLE_QUESTIONS` list at the top of the file to match the kinds of questions you actually care about.

Key tools used: `openai` as the judge model, same `RAGPipeline` from `rag.py`.

```powershell
python src\faithfulness.py
```

---

## 4. `src/api.py` — exposes the pipeline as a web server

This is an optional script that wraps the RAG pipeline in an HTTP API using **FastAPI**. Once running, any application — a website, a mobile app, a frontend you build later — can ask questions by sending an HTTP request. The RAG logic is identical to what the CLI does; this file just changes how questions arrive and how answers are returned.

It includes the same rate limiting as the interactive CLI (per IP address, 10 requests per minute), reuses the input length limit from the pipeline, and automatically generates an interactive documentation page at `/docs` where you can test it in the browser.

Run the server:
```powershell
python src\api.py
```

Then send a question:
```
POST http://localhost:8080/ask
Body: {"query": "What is theta activity?"}
```

Or open `http://localhost:8080/docs` to test it interactively in the browser.

### Conversation logging

Every request is logged as a JSON file in Google Cloud Storage under `logs/YYYY-MM-DD/`. Each file contains:

```json
{
  "timestamp": "2026-06-30T14:32:01Z",
  "conversation_id": "abc-123",
  "language": "en",
  "query": "What MLOps projects has he built?",
  "answer": "...",
  "ip": "82.45.123.XX",
  "user_agent": "Mozilla/5.0 Chrome/...",
  "referrer": "https://martinantunez.com",
  "response_time_ms": 1842,
  "num_chunks": 8,
  "top_score": 0.821,
  "sources": ["documents/martin-antunez-experience.txt"],
  "token_usage": { "input_tokens": 1240, "output_tokens": 187 },
  "history_length": 2
}
```

Logging is skipped silently when `GCS_BUCKET_NAME` is not set (safe for local development).

### Privacy

- **IP anonymization** — the last octet is replaced with `XX` (e.g. `82.45.123.XX`), which still allows city-level geolocation but cannot identify a specific user.
- **Auto-deletion** — logs are automatically deleted after 90 days via a GCS lifecycle rule. Apply it once with:
  ```powershell
  gsutil lifecycle set lifecycle.json gs://your-bucket-name
  ```
- **No cookies or tracking scripts** — logging happens server-side only, on each request.

---

## 5. `src/bot.py` — exposes the pipeline as a Telegram bot

This is an optional script that wraps the RAG pipeline as a Telegram bot using **python-telegram-bot**. Once running, you can ask questions directly from the Telegram app on your phone or desktop and get answers back as messages. Again, the RAG logic is the same — only the interface changes.

Rate limiting is applied per Telegram user (10 questions per minute), and the same input length limit from the pipeline applies.

Setup requires:
1. Creating a bot via Telegram's `@BotFather` to get a token
2. Adding the token to your `.env` file:
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   ```
3. Installing the extra package:
   ```powershell
   pip install python-telegram-bot
   ```

Run the bot:
```powershell
python src\bot.py
```

Then find your bot on Telegram and start sending questions.

---

## 6. Deploying to Google Cloud Run

The API is containerized with Docker and deployed to Cloud Run. The ChromaDB index is bundled inside the image at build time, so you must run `ingest.py` locally before building.

You only need to rebuild and redeploy when you change code or documents. If you only change documents, run `ingest.py` first to rebuild the index, then follow the steps below.

**Step 1 — Build and push the Docker image:**
```powershell
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO/rag-api "path\to\project"
```

**Step 2 — Deploy to Cloud Run:**
```powershell
gcloud run deploy rag-api `
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO/rag-api:latest `
  --region us-central1 `
  --platform managed `
  --allow-unauthenticated `
  --set-secrets OPENAI_API_KEY=openai-api-key:latest `
  --set-env-vars GCS_BUCKET_NAME=your-bucket-name
```

**Step 3 — Apply GCS log retention (run once):**
```powershell
gsutil lifecycle set lifecycle.json gs://your-bucket-name
```

The `OPENAI_API_KEY` is injected at runtime from Google Secret Manager — never baked into the image. The `GCS_BUCKET_NAME` variable tells the API where to write conversation logs; omit it to disable logging.

---

## Tests

The test suite covers the core logic — chunking, PDF cleaning, metadata extraction, source formatting — without making any API calls, so it runs offline and for free.

```powershell
python -m pytest tests/ -v --basetemp=".\tmp_pytest"
```
