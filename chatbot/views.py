import os
from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import faiss
import tiktoken
from groq import Groq
from sentence_transformers import SentenceTransformer

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', 'fallback-key')
MODEL_NAME = "llama-3.1-8b-instant"
TEXT_FILE = "chatbot/data.txt"

client = Groq(api_key=GROQ_API_KEY)
tokenizer = tiktoken.get_encoding("cl100k_base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Lazy load data
_data_loaded = False
_chunks = None
_index = None
_chunk_lookup = None

def _load_data():
    global _data_loaded, _chunks, _index, _chunk_lookup
    if _data_loaded:
        return
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        website_text = f.read()
    
    def chunk_text(text, max_tokens=300):
        tokens = tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk = tokens[i:i+max_tokens]
            chunks.append(tokenizer.decode(chunk))
        return chunks
    
    _chunks = chunk_text(website_text)
    embeddings = embedder.encode(_chunks).tolist()
    
    dimension = len(embeddings[0])
    _index = faiss.IndexFlatL2(dimension)
    _index.add(np.array(embeddings).astype("float32"))
    _chunk_lookup = {i: chunk for i, chunk in enumerate(_chunks)}
    _data_loaded = True

def ask_bot(query, top_k=3):
    _load_data()
    query_embedding = embedder.encode([query])[0]
    D, I = _index.search(np.array([query_embedding]).astype("float32"), top_k)
    valid_indices = [i for i in I[0] if i != -1]
    if not valid_indices:
        return "Sorry, I couldn't find relevant information."
    relevant_chunks = "\n\n".join(_chunk_lookup[i] for i in valid_indices)
    prompt = f"""You are a helpful assistant. Answer the question using the context below.\n\nContext:\n{relevant_chunks}\n\nQuestion: {query}\nAnswer:"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def ask_bot(query, top_k=3):
    query_embedding = embedder.encode([query])[0]
    D, I = index.search(np.array([query_embedding]).astype("float32"), top_k)
    valid_indices = [i for i in I[0] if i != -1]
    if not valid_indices:
        return "Sorry, I couldn't find relevant information."
    relevant_chunks = "\n\n".join(chunk_lookup[i] for i in valid_indices)
    prompt = f"""You are a helpful assistant. Answer the question using the context below.\n\nContext:\n{relevant_chunks}\n\nQuestion: {query}\nAnswer:"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def chatbot_view(request):
    return render(request, "chatbot/chatbot.html")

def get_answer(request):
    user_input = request.GET.get("query", "")
    if not user_input:
        return JsonResponse({"answer": "Please enter a valid query."})
    answer = ask_bot(user_input)
    return JsonResponse({"answer": answer})
