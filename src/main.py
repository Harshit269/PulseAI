import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from pipeline import run_pipeline

app = FastAPI(
    title="Pulse AI",
    description="AI-powered symptom-based medical insight chatbot using RAG.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


class HistoryTurn(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[list[HistoryTurn]] = []


class ChatResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Pulse AI", "version": "2.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Convert pydantic models to plain dicts for the pipeline
    history = [{"role": t.role, "content": t.content} for t in (request.history or [])]

    try:
        result = run_pipeline(request.message.strip(), history=history)
        return ChatResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
