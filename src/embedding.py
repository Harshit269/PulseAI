import re
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

def create_embedding(text):
    return model.encode([text])[0]

def is_valid_query(query):
    query = query.lower().strip()
    if len(query) < 5:
        return False
    if not re.search("[a-zA-Z]", query):
        return False
    if len(query.split()) < 2:
        return False
    medical_keywords = [
        "fever","cough","pain","headache","nausea",
        "vomiting","fatigue","chills","rash",
        "breathing","weakness","tiredness","muscle"
    ]
    for word in medical_keywords:
        if word in query:
            return True
    return False
