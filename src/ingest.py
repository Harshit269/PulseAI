import os
import time
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
from dotenv import load_dotenv
from embedding import create_embedding

load_dotenv()

PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pulse-ai"
NAMESPACE = "pubmed-data"

def setup_pinecone():
    pc = Pinecone(api_key=PINECONE_KEY)
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(10)
    return pc.Index(INDEX_NAME)

def ingest_data():
    index = setup_pinecone()
    dataset = load_dataset("pubmed_qa", "pqa_unlabeled", split="train")
    batch_size = 50

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        vectors = []

        for j in range(len(batch["pubid"])):
            question = batch["question"][j]
            context = " ".join(batch["context"][j]["contexts"]).strip()
            text_to_embed = f"{question} {context}"

            try:
                embedding = create_embedding(text_to_embed)
                vectors.append({
                    "id": str(batch["pubid"][j]),
                    "values": embedding,
                    "metadata": {
                        "title": question,
                        "context": context[:1000],
                        "source": "pubmed"
                    }
                })
            except Exception as e:
                print(f"Skipping record {batch['pubid'][j]}: {e}")
                continue

        if vectors:
            index.upsert(vectors=vectors, namespace=NAMESPACE)
            print(f"Upserted batch starting at {i}")
        time.sleep(0.5)

if __name__ == "__main__":
    ingest_data()