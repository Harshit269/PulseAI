import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)
_headers = {"Authorization": f"Bearer {HF_API_KEY}"}


def create_embedding(text: str) -> list[float]:
    """Return a 384-dim embedding vector for the given text."""
    response = requests.post(
        API_URL,
        headers=_headers,
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Embedding API error {response.status_code}: {response.text}")
    return response.json()
