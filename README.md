# Resilient RAG Ingestion Service

A Streamlit app for uploading PDF documents into a Pinecone vector database. Re-uploading the
same document (even under a different filename) is detected automatically — this is the
**idempotency** check — so nothing gets embedded or stored twice. Every step of the upload
(extract → chunk → embed → store) is timed and sent to **Langfuse**, so you can see exactly how
long each step took for every document you upload.

You'll need your own API keys to run this:
- A **Pinecone** API key (free tier works) — this is your vector database.
- An **OpenAI** API key — used to turn document text into embeddings.
- A **Langfuse** API key (free tier works) — this is what shows you the timing dashboard.

---

## 1. Clone the repo

```bash
git clone <this-repo-url>
cd Doc-upload
```

## 2. Set up a Python environment

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Add your API keys

Copy the example env file:

```bash
copy .env.example .env
```

Open `.env` and replace every blank/dummy value with your real keys:

```
PINECONE_API_KEY=your-real-pinecone-key
PINECONE_INDEX_NAME=test-sample-data
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_EMBEDDING_DIMENSION=3072

OPENAI_API_KEY=your-real-openai-key
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-4o-mini

LANGFUSE_PUBLIC_KEY=your-real-langfuse-public-key
LANGFUSE_SECRET_KEY=your-real-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

Where to get each key:
- **Pinecone**: [app.pinecone.io](https://app.pinecone.io) → sign up free → API Keys. You don't
  need to create the index yourself — the app creates it automatically on first run, using the
  `PINECONE_INDEX_NAME` / `PINECONE_EMBEDDING_DIMENSION` you set above.
- **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Langfuse**: [cloud.langfuse.com](https://cloud.langfuse.com) → sign up free → create a
  project → Settings → API Keys → Create new API keys. **Important**: Langfuse Cloud has multiple
  regions (e.g. `cloud.langfuse.com`, `us.cloud.langfuse.com`, `jp.cloud.langfuse.com`) — check
  your browser's address bar while logged into your project and set `LANGFUSE_HOST` to match
  exactly, or authentication will fail even with correct keys.

## 5. Run it — one command

```bash
venv\Scripts\python run.py
```

This starts both the backend (`http://127.0.0.1:8000`) and the frontend
(`http://127.0.0.1:8501`) together, in the right order. Press `Ctrl+C` once to stop both.

<details>
<summary>Alternative: run backend and frontend separately (two terminals)</summary>

Terminal 1:
```bash
venv\Scripts\python -m app.main
```

Terminal 2:
```bash
venv\Scripts\streamlit run frontend/app.py
```

</details>

## 6. Use it

1. On the Streamlit page, upload a PDF and click **Ingest**.
2. If it's a document you haven't uploaded before, you'll see: *"Ingested as `<doc_id>` — N
   chunks."*
3. If you upload the **exact same document again** (even renamed), you'll see: *"Already
   ingested as `<doc_id>` — skipped re-embedding."* This is the idempotency check working — it's
   comparing the document's content, not its filename.

---

## How Langfuse fits in

Every time you upload a document from the Streamlit app, the backend records one **trace** in
Langfuse named `ingest_document`, made up of four timed steps (spans):

| Step | What it measures |
|---|---|
| `extract_pdf` | Time to read the PDF and pull out text + tables |
| `chunk_pages` | Time to split the text into chunks, and how many chunks/tables were produced |
| `embed_chunks` | Time for OpenAI to turn those chunks into vectors |
| `upsert_vectors` | Time to write those vectors into Pinecone |

To see it: log into your Langfuse project (the host from your `.env`'s `LANGFUSE_HOST`), and go
to the **Tracing** tab in the left sidebar — not "Dashboards", which only shows aggregated
summaries. Each upload shows up as its own trace — click into one to see the breakdown of which
step took the longest, how many chunks/tables were found, and whether the upload was a fresh
ingest or a duplicate that got skipped.

If `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are left blank in `.env`, the app still works
normally — you just won't get the timing dashboard.

---

## Run the tests

```bash
venv\Scripts\python -m pytest tests/ -v
```

- `tests/unit` — pure functions (chunker, dedup hash, chunk metadata), no network calls.
- `tests/integration` — full `/ingest` and `/query` HTTP flow, with the embedder, vector store,
  and LLM swapped for mocks (no real API calls).
- `tests/resilience` — proves retries recover from transient failures, and that the circuit
  breaker fails fast after repeated failures instead of retrying forever.

## What's real vs. mocked

- **Real** (when you run the app normally): Pinecone, OpenAI embeddings + chat, PDF parsing.
- **Mocked** (only inside `tests/`): embedder, vector store, and LLM client — so tests run
  offline and don't cost API credits.

## How chunking works

- Each PDF page is parsed with `pdfplumber`. Tables are detected and pulled out separately from
  the surrounding paragraph text.
- Paragraph text is split into fixed-size chunks (1000 characters, 150 character overlap).
- Each detected table becomes its own chunk instead of being split — this stops a table row from
  getting cut in half by the fixed-size splitter.
- Idempotency check: the full document text is normalized (whitespace collapsed, lowercased) and
  hashed with `sha256`. That hash — not the filename — is what determines whether a document has
  already been ingested.

## Not built (out of scope for this project)

- Authentication, rate limiting.
- OCR for scanned/image-only PDFs — these are rejected with a clear error rather than silently
  producing empty chunks.
