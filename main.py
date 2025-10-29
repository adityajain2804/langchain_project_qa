# main.py
import warnings
import os
import subprocess
import random
import json
from pathlib import Path
from embedding_creator import create_embeddings, create_vector_on_demand

warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.*")
warnings.filterwarnings("ignore", message="None of PyTorch, TensorFlow >= 2.0, or Flax have been found.*")

from contextlib import redirect_stdout, redirect_stderr
with open(os.devnull, 'w') as _devnull:
    with redirect_stdout(_devnull), redirect_stderr(_devnull):
        from pdf_loader import load_pdf
        from qa_chain import create_qa_chain, LLM_MODEL_NAME, OLLAMA_API_BASE_URL


def get_random_response(user_input):
    greetings = ["hello", "hi", "hey", "hola", "yo", "good morning", "good evening"]
    smalltalk = ["how are you", "what's up", "who are you", "tell me about yourself"]

    user_input_lower = user_input.lower()

    if any(word in user_input_lower for word in greetings):
        return random.choice([
            "Hey there! 👋 How’s it going?",
            "Hi! Ready to explore your PDF?",
            "Hello again! What shall we look at today?",
            "Hey! 😊 Let’s dive into your document.",
        ])
    elif any(phrase in user_input_lower for phrase in smalltalk):
        return random.choice([
            "I'm just a bunch of Python code — but feeling awesome today 😄",
            "All systems online and ready!",
            "I’m good! Let’s find some answers in your PDF?",
        ])
    return None


def main():
    path_exists = os.path.exists
    path_expanduser = os.path.expanduser

    pdf_path = r"D:\Aditya_projects\langchain_pdf_qa\AI-NOTES-UNIT-1.pdf"

    if not path_exists(path_expanduser(pdf_path)):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    else:
        print(f"[INFO] Using PDF file: {pdf_path}")

    docs = load_pdf(pdf_path)
    vector_store = None
    qa_chain = None

    # ✅ Add smart FAISS cache check
    INDEX_DIR = Path("faiss_index")
    META_FILE = Path("faiss.db")

    if META_FILE.exists() and INDEX_DIR.exists():
        try:
            meta = json.loads(META_FILE.read_text())
            if (
                meta.get("embed_model") == "mxbai-embed-large"
                and meta.get("source_path") == str(pdf_path)
            ):
                print("[INFO] Found existing FAISS index. Loading from disk...")
                from langchain_ollama import OllamaEmbeddings
                from langchain_community.vectorstores import FAISS

                embeddings = OllamaEmbeddings(
                    model="mxbai-embed-large",
                    base_url="http://localhost:11434"
                )
                vector_store = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

            else:
                print("[INFO] FAISS index mismatch or missing — creating new embeddings...")
                vector_store = create_vector_on_demand(docs, source_path=pdf_path)
        except Exception as e:
            print(f"[WARN] Could not load existing FAISS index: {e}")
            vector_store = create_vector_on_demand(docs, source_path=pdf_path)
    else:
        print("[INFO] No existing FAISS index found — creating embeddings...")
        vector_store = create_vector_on_demand(docs, source_path=pdf_path)

    # ✅ Create QA chain only once
    qa_chain, _, _ = create_qa_chain(vector_store)

    # Optional: Verify Ollama setup
    print(f"[INFO] Using Ollama HTTP API base URL: {OLLAMA_API_BASE_URL}")
    try:
        import requests
        r = requests.get(OLLAMA_API_BASE_URL, timeout=3)
        print(f"[INFO] Ollama endpoint reachable (status {r.status_code})")
    except Exception as e:
        print(f"[WARN] Could not reach Ollama API: {e}")

    try:
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
        out = (res.stdout or "") + (res.stderr or "")
        if LLM_MODEL_NAME in out:
            print(f"[INFO] Ollama model '{LLM_MODEL_NAME}' found locally.")
        else:
            print(f"[WARN] Ollama model '{LLM_MODEL_NAME}' not found. Run: ollama pull {LLM_MODEL_NAME}")
    except Exception as e:
        print(f"[WARN] Could not verify Ollama model: {e}")

    help_text = """
Tips for better answers:
- Be specific in your questions
- Ask about one topic at a time
- For definitions, include 'what is' or 'define'
- For processes, ask 'how to' or 'steps for'
- Type 'exit' to quit
"""

    print("\n[READY] Ask questions about the PDF content (type 'exit' to quit, 'help' for tips):")

    while True:
        query = input("\nQuestion: ").strip()
        if not query:
            continue

        rand_reply = get_random_response(query)
        if rand_reply:
            print(f"\nAnswer: {rand_reply}\n")
            continue

        query_lower = query.lower()
        if query_lower == "exit":
            print("\nThank you for using the PDF QA system!")
            break
        if query_lower == "help":
            print(help_text)
            continue

        # 🧠 Run QA chain with timeout
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            with ThreadPoolExecutor() as executor:
                future = executor.submit(qa_chain.invoke, query)
                try:
                    answer = future.result(timeout=30)
                except TimeoutError:
                    print("\n[ERROR] Question took too long to answer. Try a more specific question.\n")
                    continue

            answer = answer.strip()
            if not answer or answer.lower() == query_lower:
                print("\nAnswer: I couldn't find relevant information in the PDF.\n")
            else:
                if len(answer) > 1500:
                    answer = answer[:1500].rsplit(".", 1)[0] + "..."
                print(f"\nAnswer: {answer}\n")

        except Exception as e:
            print(f"\n[ERROR] Failed to get answer: {e}\nTry rephrasing your question.\n")


if __name__ == "__main__":
    main()
