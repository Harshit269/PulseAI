# Pulse AI

AI-powered symptom analysis chatbot using Retrieval-Augmented Generation and medical research literature.

---

## Overview

Pulse AI is a cloud-native web application that accepts natural language descriptions of medical symptoms and returns context-aware health guidance derived from PubMed research papers. It uses a Retrieval-Augmented Generation (RAG) architecture — converting symptoms into semantic embeddings, retrieving relevant medical literature from a vector database, and generating informed responses through a large language model.

The system is fully API-driven. No heavy models run locally or on the server; all inference is delegated to managed cloud services.

---

## Architecture

```
User Input (symptoms)
  → Query validation
  → Embedding via HuggingFace Inference API (all-MiniLM-L6-v2)
  → Semantic search in Pinecone (PubMed vectors)
  → Top-k medical contexts retrieved
  → Groq API (LLaMA 3) generates response
  → Response returned to browser
```

---

## Tech Stack

| Layer            | Technology                                      |
|------------------|-------------------------------------------------|
| Web Framework    | FastAPI                                         |
| Frontend         | HTML, CSS, JavaScript (no frameworks)           |
| Embeddings       | HuggingFace Inference API — all-MiniLM-L6-v2   |
| Vector Database  | Pinecone (Serverless, AWS us-east-1)            |
| LLM              | Groq API — LLaMA 3 (llama3-8b-8192)            |
| Dataset          | PubMed QA (via HuggingFace Datasets)           |
| Deployment       | Render (web service)                            |

---

## Project Structure

```
pulse-ai/
├── src/
│   ├── main.py          # FastAPI server — serves frontend and /chat endpoint
│   ├── pipeline.py      # RAG pipeline — retrieve context, generate response
│   ├── embedding.py     # HuggingFace embedding API + query validation
│   └── ingest.py        # One-time script: load PubMed data into Pinecone
├── frontend/
│   ├── index.html       # Chatbot UI
│   ├── style.css        # Dark-mode design system
│   └── script.js        # Chat logic, API calls, animations
├── Procfile             # Render start command
├── requirements.txt
├── .env                 # API keys (never commit this)
└── .gitignore
```

---

## Local Setup (Development Only)

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/pulse-ai.git
cd pulse-ai
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
HF_API_KEY=your_huggingface_key
```

### 3. Ingest PubMed data into Pinecone (one-time)

This step loads the PubMed QA dataset and uploads vector embeddings to Pinecone. Run it once before starting the server.

```bash
python src/ingest.py
```

### 4. Start the server

```bash
uvicorn src.main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## Deploying to Render

Render is a cloud platform that can host the FastAPI backend for free.

### Steps

1. Push the project to a GitHub repository.
2. Go to [render.com](https://render.com) and create a new **Web Service**.
3. Connect your GitHub repository.
4. Set the following configuration:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
5. Add your API keys as **Environment Variables** in the Render dashboard:
   - `PINECONE_API_KEY`
   - `GROQ_API_KEY`
   - `HF_API_KEY`
6. Deploy. Render will assign a public URL (e.g., `https://pulse-ai.onrender.com`).

The frontend is served directly by the FastAPI app, so no separate frontend hosting is needed.

---

## API Reference

### `GET /`
Serves the chatbot web interface.

### `GET /health`
Returns service status.

```json
{ "status": "ok", "service": "Pulse AI" }
```

### `POST /chat`
Runs the RAG pipeline and returns a medical insight.

**Request body:**
```json
{ "message": "I have a persistent cough, fever, and fatigue" }
```

**Response:**
```json
{ "response": "Based on the retrieved medical literature, these symptoms may be associated with..." }
```

---

## Data Ingestion Details

The `ingest.py` script loads the `pubmed_qa` dataset (pqa_unlabeled split) from HuggingFace, generates 384-dimensional embeddings for each record using the HuggingFace Inference API, and upserts them into a Pinecone index named `pulse-ai` under the namespace `pubmed-data`.

This script is run once and does not need to be run again unless the Pinecone index is reset.

---

## Disclaimer

Pulse AI is built for educational and informational purposes only. It does not provide medical diagnosis and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a licensed healthcare professional regarding any medical condition or symptom. In case of a medical emergency, contact emergency services immediately.

---

## Author

Harshit