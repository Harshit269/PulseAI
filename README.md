# Pulse AI

An AI-powered medical symptom analysis chatbot that uses Retrieval-Augmented Generation (RAG) to provide context-aware health guidance based on PubMed research literature.

## What I Built

- **RAG Pipeline**: Implemented a medical retrieval pipeline that converts user symptoms into semantic embeddings using HuggingFace (`all-MiniLM-L6-v2`), retrieves relevant PubMed literature from a Pinecone vector database, and generates informed responses using the Groq API (`llama-3.1-8b-instant`).
- **Data Ingestion**: Created an ingestion script (`src/ingest.py`) to process the HuggingFace `pubmed_qa` dataset and populate the Pinecone vector index.
- **FastAPI Backend**: Built a fully API-driven FastAPI server (`src/main.py`) to manage query validation, embedding, retrieval, and LLM generation.
- **Frontend Interface**: Designed and developed a clean web interface (HTML/CSS/JS) for seamless user interaction.
- **Deployment**: Configured and deployed the application as a cloud-native web service on Render.

## Tech Stack

- **Backend**: Python, FastAPI
- **AI & Data**: HuggingFace Inference API, Pinecone, Groq (Llama 3)
- **Frontend**: HTML, CSS, Vanilla JavaScript
- **Deployment**: Render

## Project Summary

- `src/main.py`: FastAPI server and endpoints.
- `src/pipeline.py` & `src/embedding.py`: RAG orchestrator and embedding logic.
- `src/ingest.py`: Script to embed and upsert PubMed data to Pinecone.
- `frontend/`: UI files (`index.html`, `style.css`, `script.js`).

## How to Run Locally

1. Install dependencies: `pip install -r requirements.txt`
2. Create a `.env` file with `PINECONE_API_KEY`, `GROQ_API_KEY`, and `HF_API_KEY`.
3. run data ingestion once: `python src/ingest.py`
4. Start the server: `uvicorn src.main:app --reload`
5. Access the web interface at `http://localhost:8000`.