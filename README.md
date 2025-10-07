# 🧠 My Chatbot (Django + Groq + FAISS)

This project is a chatbot built with **Django**, **Groq API**, **FAISS**, and **Sentence Transformers**.  
It indexes text data, retrieves the most relevant chunks using embeddings, and then generates natural language answers with a Groq LLM.

---

## 🚀 Features
- Text retrieval using **FAISS vector search**.
- Embeddings generated with **Sentence Transformers** (`all-MiniLM-L6-v2`).
- Context-aware responses powered by **Groq LLMs**.
- Simple **Django web UI** (`chatbot.html`) to interact with the bot.

---

## 📦 Requirements
- Python 3.9+
- Virtual environment (recommended)
- Groq API key

Install dependencies:

```bash
pip install -r requirements.txt
