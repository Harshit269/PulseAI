import os
import requests
import re
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def create_embedding(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}})
    if response.status_code != 200:
        raise Exception(f"Embedding API failed: {response.text}")
    return response.json()

def is_valid_query(query):
    query = query.lower().strip()
    if len(query) < 5 or not re.search("[a-zA-Z]", query) or len(query.split()) < 2:
        return False
    
    medical_keywords = {
        "fever", "cough", "pain", "headache", "nausea", "vomiting", 
        "fatigue", "chills", "rash", "breathing", "weakness", "tiredness", "muscle"
    }
    return any(word in query for word in medical_keywords)