import os
import faiss
from sentence_transformers import SentenceTransformer
from chunking import load_text, chunk_text
import re

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search(query, index, chunks, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    results = []
    for i in indices[0]:
        results.append(chunks[i])

    return results

def is_valid_query(query):
    query = query.lower().strip()

    # basic checks
    if len(query) < 5:
        return False

    if not re.search("[a-zA-Z]", query):
        return False

    if len(query.split()) < 2:
        return False

    # medical keywords check
    medical_keywords = [
        "fever", "cough", "pain", "headache", "nausea",
        "vomiting", "fatigue", "chills", "rash",
        "breathing", "weakness", "tiredness", "muscle"
    ]

    # check if at least one keyword present
    for word in medical_keywords:
        if word in query:
            return True

    return False

def main():
    file_path = os.path.join("data", "sample.txt")

    # Load and chunk data
    text = load_text(file_path)
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    # Create embeddings
    embeddings = create_embeddings(chunks)

    # Create FAISS index
    index = create_faiss_index(embeddings)

    print("FAISS index created successfully!\n")

    query = input("Enter symptoms: ")

    if not is_valid_query(query):
        print("\n⚠️ Invalid input. Please enter meaningful symptoms (e.g., 'fever and headache').")
        return
    
    results = search(query, index, chunks)

    print("\nTop matching medical information:\n")
    for i, res in enumerate(results):
        print(f"{i+1}. {res}\n")


if __name__ == "__main__":
    main()
