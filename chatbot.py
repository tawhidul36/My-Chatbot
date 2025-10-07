import numpy as np
import faiss
import tiktoken
from groq import Groq
from sentence_transformers import SentenceTransformer


GROQ_API_KEY = "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"
MODEL_NAME = "llama3-8b-8192"
TEXT_FILE = "data.txt"


client = Groq(api_key=GROQ_API_KEY)
tokenizer = tiktoken.get_encoding("cl100k_base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, max_tokens=300):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i+max_tokens]
        chunks.append(tokenizer.decode(chunk))
    return chunks

with open(TEXT_FILE, "r", encoding="utf-8") as f:
    website_text = f.read()

chunks = chunk_text(website_text)


def get_embeddings(texts):
    return embedder.encode(texts).tolist()

embeddings = get_embeddings(chunks)


dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))
chunk_lookup = {i: chunk for i, chunk in enumerate(chunks)}

def ask_bot(query, top_k=3):
    query_embedding = get_embeddings([query])[0]
    D, I = index.search(np.array([query_embedding]).astype("float32"), top_k)

   
    valid_indices = [i for i in I[0] if i != -1]
    if not valid_indices:
        return "Sorry, I couldn't find relevant information."

    relevant_chunks = "\n\n".join(chunk_lookup[i] for i in valid_indices)
    prompt = f"""You are a helpful assistant. Answer the question using the context below.

Context:
{relevant_chunks}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    while True:
        user_q = input("\nAsk a question (or type 'exit'): ")
        if user_q.lower() == 'exit':
            break
        answer = ask_bot(user_q)
        print("\n🤖 Bot:", answer)
