# Pulse AI 🫀

> An AI-powered medical research assistant that uses Retrieval-Augmented Generation (RAG) to provide symptom-based health guidance grounded in PubMed literature.

---

## What It Does

Pulse AI lets you describe symptoms in plain language and get informed, research-backed responses drawn from PubMed — the largest database of biomedical literature. It's not a diagnosis tool. It's a research companion that helps you understand what the literature says about what you're experiencing, and connects you with a doctor to follow up.

---

## Architecture

```
User Message
     │
     ▼
┌─────────────────────────────────────────┐
│           Intent Classifier             │
│     llama-3.1-8b-instant (Groq)         │
│     MEDICAL  ──────────►  CHAT          │
└──────────┬──────────────────────────────┘
           │ MEDICAL
           ▼
┌─────────────────────────────────────────┐
│           Embedding (HuggingFace)        │
│        all-MiniLM-L6-v2  (384-dim)      │
│    embeds query only — not context      │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│         Pinecone Vector Search          │
│     top-10 candidates retrieved         │
│     cosine similarity, filtered >0.3   │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│           LLM Reranker                  │
│     llama-3.1-8b-instant scores each   │
│     chunk 0-10 → top 3 selected        │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│        Response Generation              │
│    llama-3.3-70b-versatile (Groq)       │
│    streamed via SSE                     │
│    full conversation history injected  │
└──────────┬──────────────────────────────┘
           │
           ▼
     Streaming Response
     + Source citations
     + Doctor suggestion
```

---

## Features

**RAG Pipeline**
- Embeds user queries using `all-MiniLM-L6-v2` via the HuggingFace Inference API
- Retrieves top-10 PubMed candidates from a Pinecone vector index
- Reranks using `llama-3.1-8b-instant` in a single prompt call (no local model download, zero cold-start delay)
- Passes the best 3 chunks as grounded context to the generation model

**Conversational AI**
- Powered by `llama-3.3-70b-versatile` on Groq — significantly more capable and natural than smaller models
- Intent classifier routes messages: medical queries trigger RAG, casual messages skip it entirely and just converse
- Full conversation history sent on every request so the model remembers what was said
- Warm, human-sounding personality — greets, follows up, and answers naturally

**Streaming**
- Responses stream token-by-token via Server-Sent Events (SSE)
- Blinking cursor while streaming, instant feel even for long answers

**Multi-conversation Sidebar**
- Create, switch between, and delete multiple conversations
- Conversations grouped by Today / Yesterday / Previous 7 days / Previous 30 days
- Auto-titled from the first message
- Search bar to filter conversations
- Persisted to `localStorage` — survives page refresh

**Source Citations**
- PubMed papers used to generate each response shown as clickable chips below the message
- Direct links to `pubmed.ncbi.nlm.nih.gov`

**Doctor Suggestion**
- Every medical response includes a doctor contact card with a Call button
- On mobile: taps straight to the phone dialer
- On desktop: shows a popup with the number and a copy button

**Link Preview Panel**
- Paste any URL in chat — a side panel slides in with the domain, favicon, and an AI-generated summary
- Panel is dismissible and doesn't interrupt the conversation

**Mobile UI**
- Full responsive layout — sidebar becomes a slide-in drawer with a hamburger button
- Safe area insets for iPhone notches and home indicator
- 16px input font to prevent iOS auto-zoom
- Touch-optimised tap targets throughout

**Production-ready**
- Rate limiting via `slowapi` (20 req/min per IP)
- Startup validation — fails immediately with a clear message if any API key is missing
- Input length capped at 1000 characters (Pydantic `Field`)
- Per-step error handling for embedding, retrieval, and generation — each fails gracefully

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| LLM | Groq API — `llama-3.3-70b-versatile` (generation), `llama-3.1-8b-instant` (classifier + reranker) |
| Embeddings | HuggingFace Inference API — `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | Pinecone (serverless, cosine similarity) |
| Data | HuggingFace `pubmed_qa` dataset (pqa_unlabeled split) |
| Frontend | Vanilla HTML, CSS, JavaScript — no framework |
| Deployment | Render |

---

## Project Structure

```
PulseAI/
├── src/
│   ├── main.py          # FastAPI app — endpoints, rate limiting, SSE streaming
│   ├── pipeline.py      # RAG orchestrator — classify → embed → retrieve → rerank → generate
│   ├── embedding.py     # HuggingFace embedding API wrapper
│   └── ingest.py        # One-time script to embed and upsert PubMed data to Pinecone
├── frontend/
│   └── index.html       # Full UI — multi-chat sidebar, streaming, mobile layout
├── requirements.txt
├── .env.example
└── README.md
```

---

## How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/Harshit269/PulseAI.git
cd PulseAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
HF_API_KEY=your_huggingface_key

**Get your keys:**
- Pinecone → [pinecone.io](https://www.pinecone.io/) (free tier available)
- Groq → [console.groq.com](https://console.groq.com/) (free tier, fast inference)
- HuggingFace → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (free)

### 4. Ingest PubMed data (run once)

This downloads the `pubmed_qa` dataset and populates your Pinecone index. Takes a few minutes depending on your connection.

```bash
python src/ingest.py
```

> Note: if you've run ingest before with an older version of the code, run it again — the new version embeds only the question (not the context blob), which gives sharper retrieval.

### 5. Start the server

```bash
uvicorn src.main:app --reload
```

### 6. Open the app

Visit [http://localhost:8000](http://localhost:8000)

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `GROQ_API_KEY` | Yes | Groq API key |
| `HF_API_KEY` | Yes | HuggingFace API key |
| `DOCTOR_NAME` | No | Doctor's name shown in chat card |
| `DOCTOR_PHONE` | No | Phone number for the call button |
| `DOCTOR_SPECIALTY` | No | Doctor's specialty |
| `DOCTOR_NOTE` | No | Availability note shown on card |

---

## Deployment on Render

The app is deployed as a web service on Render. To deploy your own:

1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
5. Add all environment variables in the Render dashboard
6. Run `python src/ingest.py` once manually or as a one-off job

---

## Important Disclaimer

Pulse AI is an informational tool only. It surfaces relevant research from PubMed based on described symptoms. It does **not** provide a medical diagnosis and should never be used as a substitute for consultation with a licensed healthcare professional. In an emergency, contact emergency services immediately.
