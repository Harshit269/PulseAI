import os
import sys
import json
import logging
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pipeline import stream_pipeline, run_pipeline

logger = logging.getLogger(__name__)

# ── Startup: validate required env vars ───────────────────────────────────────
REQUIRED_ENV_VARS = ["PINECONE_API_KEY", "GROQ_API_KEY", "HF_API_KEY"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Please set them in your .env file or deployment config."
        )
    logger.info("All required environment variables present. Pulse AI starting.")
    yield
    logger.info("Pulse AI shutting down.")


# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(
    title="Pulse AI",
    description="AI-powered symptom-based medical insight chatbot using RAG.",
    version="2.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# ── Schema ─────────────────────────────────────────────────────────────────────
class HistoryTurn(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    history: Optional[list[HistoryTurn]] = []

class ChatResponse(BaseModel):
    response: str


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Pulse AI", "version": "2.1.0"}


@app.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(request: Request, body: ChatRequest):
    """
    SSE streaming endpoint. Each event is a JSON line:
      data: {"type": "text",    "content": "..."}
      data: {"type": "sources", "content": [...]}
      data: {"type": "warning", "content": "..."}
      data: [DONE]
    """
    history = [{"role": t.role, "content": t.content} for t in (body.history or [])]

    def generate():
        try:
            for event in stream_pipeline(body.message.strip(), history):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Unexpected streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'An unexpected error occurred.'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # important for Nginx/Render to not buffer SSE
        },
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest):
    """Non-streaming fallback — used for link summaries."""
    history = [{"role": t.role, "content": t.content} for t in (body.history or [])]
    try:
        result = run_pipeline(body.message.strip(), history=history)
        return ChatResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
