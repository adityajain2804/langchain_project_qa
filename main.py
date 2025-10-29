# main.py
import warnings
import os
import random
import json
from pathlib import Path
from embedding_creator import create_vector_on_demand
from pdf_loader import load_pdf
from qa_chain import create_qa_chain, LLM_MODEL_NAME, OLLAMA_API_BASE_URL

# Suppress warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.*")
warnings.filterwarnings("ignore", message="None of PyTorch, TensorFlow >= 2.0, or Flax have been found.*")

# ✅ Global setup (only runs once)
PDF_PATH = r"D:\Aditya_projects\langchain_pdf_qa\AI-NOTES-UNIT-1.pdf"
INDEX_DIR = Path("faiss_index")
META_FILE = Path("faiss.db")

print(f"[INFO] Loading PDF from: {PDF_PATH}")
docs = load_pdf(PDF_PATH)

vector_store = None
if META_FILE.exists() and INDEX_DIR.exists():
    try:
        meta = json.loads(META_FILE.read_text())
        if meta.get("embed_model") == "mxbai-embed-large" and meta.get("source_path") == str(PDF_PATH):
            print("[INFO] Loading FAISS index from disk...")
            from langchain_ollama import OllamaEmbeddings
            from langchain_community.vectorstores import FAISS
            embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
            vector_store = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        else:
            print("[INFO] FAISS index mismatch — regenerating...")
            vector_store = create_vector_on_demand(docs, source_path=PDF_PATH)
    except Exception as e:
        print(f"[WARN] Failed to load FAISS index: {e}")
        vector_store = create_vector_on_demand(docs, source_path=PDF_PATH)
else:
    print("[INFO] No FAISS index found — creating new one...")
    vector_store = create_vector_on_demand(docs, source_path=PDF_PATH)

qa_chain, _, _ = create_qa_chain(vector_store)
print(f"[INFO] QA chain ready using model '{LLM_MODEL_NAME}'")

# 🧠 Small talk handler
def get_random_response(user_input):
    greetings = ["hello", "hi", "hey", "hola", "yo"]
    smalltalk = ["how are you", "what's up", "who are you"]
    lower = user_input.lower()

    if any(g in lower for g in greetings):
        return random.choice([
            "Hey there! 👋 How’s it going?",
            "Hi! Ready to explore your PDF?",
            "Hello! What shall we look at today?",
        ])
    elif any(s in lower for s in smalltalk):
        return random.choice([
            "I'm just a Python app, but I'm doing great 😄",
            "All systems are running smoothly!",
        ])
    return None


# ✅ Function to be used by app.py
def answer_question(question: str):
    """Takes a question and returns an answer from the PDF."""
    rand = get_random_response(question)
    if rand:
        return rand

    try:
        answer = qa_chain.invoke(question).strip()
        if not answer:
            return "Sorry, I couldn't find relevant information in the PDF."
        return answer
    except Exception as e:
        return f"Error: {str(e)}"
