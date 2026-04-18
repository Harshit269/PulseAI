import os
import re
import time
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
from dotenv import load_dotenv
from embedding import create_embedding

load_dotenv()

PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME   = "pulse-ai"
NAMESPACE    = "pubmed-data"


def clean_context(raw_contexts: list[str], max_chars: int = 1200) -> str:
    """
    Joins context passages and truncates cleanly at a sentence boundary
    rather than cutting mid-word at a fixed character count.
    """
    joined = " ".join(c.strip() for c in raw_contexts if c.strip())

    if len(joined) <= max_chars:
        return joined

    # Truncate at the last sentence boundary within the limit
    truncated = joined[:max_chars]
    # Find the last full stop, question mark, or exclamation before the limit
    match = re.search(r'[.!?][^.!?]*$', truncated)
    if match:
        truncated = truncated[:match.start() + 1]

    return truncated.strip()


def setup_pinecone():
    pc = Pinecone(api_key=PINECONE_KEY)
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        time.sleep(10)
    return pc.Index(INDEX_NAME)


def ingest_data():
    index = setup_pinecone()
    dataset = load_dataset("pubmed_qa", "pqa_unlabeled", split="train")
    batch_size = 50

    for i in range(0, len(dataset), batch_size):
        batch   = dataset[i: i + batch_size]
        vectors = []

        for j in range(len(batch["pubid"])):
            question = batch["question"][j]
            context  = clean_context(batch["context"][j]["contexts"])

            # Embed ONLY the question — keeps the vector focused on the
            # retrieval signal. Context is stored as metadata, not mixed
            # into the embedding, so similarity search stays sharp.
            try:
                embedding = create_embedding(question)
                vectors.append({
                    "id": str(batch["pubid"][j]),
                    "values": embedding,
                    "metadata": {
                        "title":   question,
                        "context": context,
                        "source":  "pubmed",
                    },
                })
            except Exception as e:
                print(f"Skipping record {batch['pubid'][j]}: {e}")
                continue

        if vectors:
            index.upsert(vectors=vectors, namespace=NAMESPACE)
            print(f"Upserted batch {i} – {i + len(vectors)}")

        time.sleep(0.5)


if __name__ == "__main__":
    ingest_data()
