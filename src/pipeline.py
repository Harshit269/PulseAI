import os
from pinecone import Pinecone
from dotenv import load_dotenv
from groq import Groq
from embedding import create_embedding, is_valid_query

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("pulse-ai")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
NAMESPACE = "pubmed-data"

def retrieve_context(query, k=3):
    embedding = create_embedding(query)
    results = index.query(
        vector=embedding,
        top_k=k,
        include_metadata=True,
        namespace=NAMESPACE
    )
    return [match["metadata"]["context"] for match in results["matches"]]

def generate_answer(query, contexts):
    context_text = "\n\n".join(contexts)
    system_prompt = "You are a helpful medical assistant AI. Provide information based on references. Do not diagnose. Do not use any markdown formatting such as bold, italics, or bullet symbols — respond in plain text only."
    user_prompt = f"""
Patient symptoms: {query}

Medical references:
{context_text}

Based on the references above, provide possible conditions and general advice. 
1. Do not provide a final diagnosis.
2. Always recommend consulting a healthcare professional.
"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.2
    )
    return response.choices[0].message.content

def run_pipeline(query):
    if not is_valid_query(query):
        return "Please enter valid medical symptoms (e.g., 'I have a persistent cough and fever')."
    
    contexts = retrieve_context(query)
    if not contexts:
        return "I couldn't find relevant medical data. Please consult a doctor."
        
    return generate_answer(query, contexts)