import os
import logging
from typing import Generator
from pinecone import Pinecone
from dotenv import load_dotenv
from groq import Groq
from embedding import create_embedding

load_dotenv()
logger = logging.getLogger(__name__)

# ── Clients ────────────────────────────────────────────────────────────────────
pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index  = pc.Index("pulse-ai")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

NAMESPACE        = "pubmed-data"
MAIN_MODEL       = "llama-3.3-70b-versatile"
CLASSIFIER_MODEL = "llama-3.1-8b-instant"   # also used for reranking — fast & free

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Pulse AI — a knowledgeable, empathetic medical research assistant with a warm and human personality.

You operate in two modes and switch between them naturally:

MEDICAL MODE — when the user describes symptoms, asks about conditions, medications, or health concerns:
  • Use the retrieved PubMed context provided to give an informed, thoughtful response.
  • When citing information, naturally reference the source in your prose — e.g. "According to a study on respiratory infections..." or "Research on this condition suggests...". Do NOT use numbered citations or footnotes.
  • Surface possible conditions and general guidance based on the literature.
  • NEVER give a definitive diagnosis. Always recommend consulting a licensed healthcare professional.
  • Acknowledge the user's concern with empathy before diving into information.

CONVERSATIONAL MODE — when the user is chatting, asking follow-ups, or being casual:
  • Respond like a warm, intelligent human companion. Match the user's energy.
  • If they say "hi" or "thanks", just respond naturally — do not force medical content.
  • If a follow-up references prior conversation, answer it directly and coherently.

Always:
  • Remember everything said in this conversation and refer back to it naturally.
  • Never use markdown formatting (no bold, no bullets, no headers) — flowing prose only.
  • Be concise when the answer is simple. Be thorough when the topic is serious.
  • Sound like a real person, not a help document."""


# ── Intent classifier ──────────────────────────────────────────────────────────
def classify_intent(message: str) -> str:
    """Returns 'MEDICAL' or 'CHAT'."""
    try:
        resp = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user message as exactly one of: MEDICAL or CHAT.\n"
                        "MEDICAL: symptoms, conditions, medications, body concerns, health questions.\n"
                        "CHAT: greetings, small talk, follow-up questions, thanks, questions about the assistant.\n"
                        "Reply with one word only: MEDICAL or CHAT."
                    ),
                },
                {"role": "user", "content": message},
            ],
            model=CLASSIFIER_MODEL,
            temperature=0,
            max_tokens=5,
        )
        label = resp.choices[0].message.content.strip().upper()
        return "MEDICAL" if "MEDICAL" in label else "CHAT"
    except Exception as e:
        logger.warning(f"Intent classification failed, defaulting to MEDICAL: {e}")
        return "MEDICAL"


# ── LLM-based reranker ─────────────────────────────────────────────────────────
def llm_rerank(query: str, matches: list, k_final: int = 3) -> list:
    """
    Uses the fast 8b model to score each retrieved chunk's relevance to the query
    in a single API call. Returns the top k_final matches sorted by score.

    Scoring prompt asks for a 0-10 integer per chunk. We parse the response and
    sort — no local model, no download, no cold-start delay.
    """
    if not matches:
        return []
    if len(matches) <= k_final:
        return matches

    # Build a numbered list of chunks for the model to score
    chunks_text = "\n\n".join(
        f"[{i+1}] {m['metadata'].get('context', '')[:400]}"
        for i, m in enumerate(matches)
    )

    prompt = (
        f"Query: {query}\n\n"
        f"Below are {len(matches)} text chunks retrieved from a medical database. "
        f"Score each chunk's relevance to the query on a scale of 0-10, "
        f"where 10 = highly relevant and 0 = completely irrelevant.\n\n"
        f"{chunks_text}\n\n"
        f"Reply with ONLY a comma-separated list of scores in order, e.g.: 8,3,7,2,9\n"
        f"No explanation, no labels — just the numbers."
    )

    try:
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise relevance scoring assistant."},
                {"role": "user", "content": prompt},
            ],
            model=CLASSIFIER_MODEL,
            temperature=0,
            max_tokens=60,
        )
        raw = resp.choices[0].message.content.strip()
        scores = [float(s.strip()) for s in raw.split(",") if s.strip().replace(".", "").isdigit()]

        if len(scores) != len(matches):
            logger.warning(f"Reranker score count mismatch ({len(scores)} vs {len(matches)}), using cosine order")
            return matches[:k_final]

        ranked = sorted(zip(scores, matches), key=lambda x: x[0], reverse=True)
        return [m for _, m in ranked[:k_final]]

    except Exception as e:
        logger.warning(f"LLM reranking failed, using cosine order: {e}")
        return matches[:k_final]


# ── Retrieval ──────────────────────────────────────────────────────────────────
def retrieve_and_rerank(query: str, k_retrieve: int = 10, k_final: int = 3):
    """
    Returns (contexts, sources).
    Embeds only the query (not context), retrieves k_retrieve candidates,
    then reranks via LLM to return the best k_final.
    """
    try:
        # Embed only the query — keeps vectors clean and focused
        embedding = create_embedding(query)
    except Exception as e:
        raise RuntimeError(f"Embedding step failed: {e}") from e

    try:
        results = index.query(
            vector=embedding,
            top_k=k_retrieve,
            include_metadata=True,
            namespace=NAMESPACE,
        )
        matches = results.get("matches", [])
    except Exception as e:
        raise RuntimeError(f"Pinecone retrieval failed: {e}") from e

    if not matches:
        return [], []

    # Filter out very low-confidence matches (cosine score < 0.3)
    matches = [m for m in matches if m.get("score", 1) >= 0.3]
    if not matches:
        return [], []

    # LLM rerank — single API call, no local model
    best = llm_rerank(query, matches, k_final=k_final)

    contexts = [m["metadata"].get("context", "") for m in best if m["metadata"].get("context")]
    sources = [
        {
            "title": m["metadata"].get("title", "PubMed article"),
            "pubid": m.get("id", ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{m.get('id', '')}/",
        }
        for m in best
    ]
    return contexts, sources


# ── Message builder ────────────────────────────────────────────────────────────
def build_messages(history: list, user_message: str, context_text: str | None = None) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-12:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    final_content = (
        f"Patient message: {user_message}\n\nRetrieved PubMed context:\n{context_text}"
        if context_text
        else user_message
    )
    messages.append({"role": "user", "content": final_content})
    return messages


# ── Streaming pipeline ─────────────────────────────────────────────────────────
def stream_pipeline(query: str, history: list | None = None) -> Generator:
    """
    Yields dicts:
      {"type": "text",    "content": "<chunk>"}
      {"type": "sources", "content": [{title, pubid, url}]}
      {"type": "warning", "content": "<message>"}
      {"type": "error",   "content": "<message>"}
    """
    if history is None:
        history = []

    if not query or not query.strip():
        yield {"type": "text", "content": "I didn't catch that — could you describe how you're feeling?"}
        return

    intent = classify_intent(query)

    context_text = None
    sources = []
    if intent == "MEDICAL":
        try:
            contexts, sources = retrieve_and_rerank(query)
            if contexts:
                context_text = "\n\n---\n\n".join(contexts)
        except RuntimeError as e:
            logger.error(f"Retrieval error: {e}")
            yield {
                "type": "warning",
                "content": "I'm having trouble searching the medical database right now, but I'll do my best with what I know.",
            }

    messages = build_messages(history, query, context_text)

    try:
        stream = client.chat.completions.create(
            messages=messages,
            model=MAIN_MODEL,
            temperature=0.6,
            max_tokens=700,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield {"type": "text", "content": delta}
    except Exception as e:
        logger.error(f"LLM streaming error: {e}")
        yield {"type": "error", "content": "Something went wrong generating a response. Please try again."}
        return

    if sources:
        yield {"type": "sources", "content": sources}


# ── Non-streaming (for link summaries) ────────────────────────────────────────
def run_pipeline(query: str, history: list | None = None) -> str:
    parts = []
    for event in stream_pipeline(query, history):
        if event["type"] == "text":
            parts.append(event["content"])
        elif event["type"] in ("error", "warning") and not parts:
            return event["content"]
    return "".join(parts).strip() or "I couldn't generate a response. Please try again."
