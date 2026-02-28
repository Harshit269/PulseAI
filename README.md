# Pulse AI

AI-powered symptom analysis chatbot using Retrieval-Augmented Generation and PubMed medical research literature.

---

## Overview

Pulse AI is a cloud-native web application that accepts natural language descriptions of medical symptoms and returns context-aware health guidance derived from PubMed research papers. It uses a Retrieval-Augmented Generation (RAG) architecture — converting symptoms into semantic embeddings, retrieving relevant medical literature from a vector database, and generating informed responses through a large language model.

The system is fully API-driven. No heavy models run locally or on the server; all inference is delegated to managed cloud services.

---

## Architecture

```
User Input
  → Query validation
  → HuggingFace Inference API (all-MiniLM-L6-v2)
  → Pinecone semantic search (PubMed vectors, top-k: 3)
  → Groq API (llama-3.1-8b-instant)
  → Response returned to browser
```

---

## Tech Stack

| Layer            | Technology                                    |
|------------------|-----------------------------------------------|
| Web Framework    | FastAPI                                       |
| Frontend         | HTML, CSS, JavaScript                         |
| Embeddings       | HuggingFace Inference API — all-MiniLM-L6-v2 |
| Vector Database  | Pinecone (Serverless, AWS us-east-1)          |
| LLM              | Groq API — llama-3.1-8b-instant              |
| Dataset          | PubMed QA (pqa_unlabeled, HuggingFace)       |
| Deployment       | Render                                        |

---

## Project Structure

```
pulse-ai/
├── src/
│   ├── main.py        # FastAPI server
│   ├── pipeline.py    # RAG pipeline
│   ├── embedding.py   # Embedding API and query validation
│   └── ingest.py      # One-time data ingestion script
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── Procfile
├── requirements.txt
├── .env
└── .gitignore
```

---

## Local Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/pulse-ai.git
cd pulse-ai
python -m venv venv
venv\Scripts\activate
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

1. Push the project to a GitHub repository.
2. Go to [render.com](https://render.com) and create a new **Web Service**.
3. Connect your GitHub repository.
4. Set the following:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables: `PINECONE_API_KEY`, `GROQ_API_KEY`, `HF_API_KEY`
6. Deploy.

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

**Request:**
```json
{ "message": "I have a persistent cough, fever, and fatigue" }
```

**Response:**
```json
{ "response": "Based on the retrieved medical literature, these symptoms may be associated with..." }
```

---

## Data Ingestion

The `ingest.py` script loads the `pubmed_qa` dataset (pqa_unlabeled split), generates 384-dimensional embeddings using the HuggingFace Inference API, and upserts them into a Pinecone index named `pulse-ai` under the namespace `pubmed-data`. This script only needs to be run once.

---

## Disclaimer

Pulse AI is built for educational and informational purposes only. It does not provide medical diagnosis and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a licensed healthcare professional. In case of a medical emergency, contact emergency services immediately.

---

## Author

Harshit