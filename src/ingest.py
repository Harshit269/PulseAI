import os
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pulse-ai"
NAMESPACE = "pubmed-data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def setup_pinecone():
    pc = Pinecone(api_key=PINECONE_KEY)
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)
    return pc.Index(INDEX_NAME)

def clean_context(context_list):
    return " ".join(context_list).strip()

def ingest_data():
    index = setup_pinecone()
    model = SentenceTransformer(MODEL_NAME)

    dataset = load_dataset("pubmed_qa", "pqa_unlabeled", split="train")

    batch_size = 100

    for i in range(0, len(dataset), batch_size):
        print(f"Uploading batch {i} to {i+batch_size}")
        batch = dataset[i:i+batch_size]

        vectors = []

        for j in range(len(batch["pubid"])):
            question = batch["question"][j]
            context = clean_context(batch["context"][j]["contexts"])

            combined_text = question + " " + context

            embedding = model.encode([combined_text])[0]

            vectors.append({
                "id": str(batch["pubid"][j]),
                "values": embedding.tolist(),
                "metadata": {
                    "title": question,
                    "context": context,
                    "source": "pubmed"
                }
            })

        index.upsert(vectors=vectors, namespace=NAMESPACE)

if __name__ == "__main__":
    ingest_data()
