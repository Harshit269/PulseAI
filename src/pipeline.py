import os
from pinecone import Pinecone
from dotenv import load_dotenv
from groq import Groq
from embedding import create_embedding

load_dotenv()

pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index  = pc.Index("pulse-ai")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

NAMESPACE        = "pubmed-data"
MAIN_MODEL       = "llama-3.3-70b-versatile"   # smart, conversational, free on Groq
CLASSIFIER_MODEL = "llama-3.1-8b-instant"       # fast, cheap — only used for intent classification

SYSTEM_PROMPT = """You are Pulse AI — a knowledgeable, empathetic medical research assistant with a warm and human personality.

You have two modes, and you switch between them naturally:

MEDICAL MODE — when the user describes symptoms, asks about conditions, medications, or health concerns:
  • Draw on the retrieved PubMed context provided to give an informed, thoughtful response.
  • Surface possible conditions and general guidance based on the literature.
  • NEVER give a definitive diagnosis. Always recommend consulting a licensed healthcare professional.
  • Acknowledge the user's concern with empathy before diving into information.

CONVERSATIONAL MODE — when the user is chatting, asking about you, following up, or saying something casual:
  • Respond like a warm, intelligent human companion would.
  • Be natural, engaging, and personable. Match the user's energy.
  • If they say "hi" or "thanks", just respond naturally — don't force medical content.
  • If a follow-up question references the previous conversation, answer it directly and coherently.

General rules that always apply:
  • Remember everything said in this conversation and refer back to it naturally.
  • Never use markdown formatting (no bold, no bullets, no headers) — respond in clean, flowing prose.
  • Be concise when the answer is simple. Be thorough when the topic is serious.
  • Sound like a real person, not a help document."""


def classify_intent(message: str) -> str:
    """
    Uses a fast small model to classify the message.
    Returns 'MEDICAL' if health-related, 'CHAT' otherwise.
    """
    try:
        resp = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier. Classify the user's message into one of two categories:\n"
                        "MEDICAL — the user is describing symptoms, a health condition, a medication, a body part concern, "
                        "or asking a health/medical question.\n"
                        "CHAT — the user is greeting, making small talk, asking about the assistant, saying thanks, "
                        "asking a follow-up question about a previous answer, or anything non-medical.\n"
                        "Reply with exactly one word: MEDICAL or CHAT. Nothing else."
                    )
                },
                {"role": "user", "content": message}
            ],
            model=CLASSIFIER_MODEL,
            temperature=0,
            max_tokens=5,
        )
        label = resp.choices[0].message.content.strip().upper()
        return "MEDICAL" if "MEDICAL" in label else "CHAT"
    except Exception:
        # If classification fails, default to MEDICAL to be safe
        return "MEDICAL"


def retrieve_context(query: str, k: int = 4) -> list[str]:
    embedding = create_embedding(query)
    results = index.query(
        vector=embedding,
        top_k=k,
        include_metadata=True,
        namespace=NAMESPACE,
    )
    return [m["metadata"]["context"] for m in results["matches"] if m.get("metadata", {}).get("context")]


def build_messages(history: list[dict], user_message: str, context_text: str | None = None) -> list[dict]:
    """
    Builds the full messages array for the LLM.
    history: list of {role: 'user'|'assistant', content: str}
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject conversation history (cap at last 12 turns to stay within context limits)
    for turn in history[-12:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    # Build the final user message, optionally with RAG context
    if context_text:
        final_user_content = (
            f"Patient message: {user_message}\n\n"
            f"Retrieved PubMed context (use this to inform your response):\n{context_text}"
        )
    else:
        final_user_content = user_message

    messages.append({"role": "user", "content": final_user_content})
    return messages


def run_pipeline(query: str, history: list[dict] | None = None) -> str:
    if history is None:
        history = []

    if not query or not query.strip():
        return "I didn't catch that — could you describe how you're feeling?"

    # Step 1: classify intent
    intent = classify_intent(query)

    # Step 2: if medical, retrieve relevant PubMed context
    context_text = None
    if intent == "MEDICAL":
        try:
            contexts = retrieve_context(query)
            if contexts:
                context_text = "\n\n---\n\n".join(contexts)
        except Exception:
            # Retrieval failed — still respond, just without RAG context
            context_text = None

    # Step 3: build message array and generate response
    messages = build_messages(history, query, context_text)

    response = client.chat.completions.create(
        messages=messages,
        model=MAIN_MODEL,
        temperature=0.6,        # Slightly higher for more natural, varied responses
        max_tokens=700,
    )

    return response.choices[0].message.content.strip()
