# main.py
import warnings
import os
import subprocess

# Suppress a known LangChain/Pydantic compatibility user warning on Python 3.14+
# This is safe to suppress locally; consider updating langchain_core or Python version later.
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.*")
# Suppress the transformers/langchain framework advisory which prints when PyTorch/TF/Flax are missing.
warnings.filterwarnings("ignore", message="None of PyTorch, TensorFlow >= 2.0, or Flax have been found.*")

from contextlib import redirect_stdout, redirect_stderr

# Some imported libraries print advisory/info messages during import (for example
# transformers/langchain may print framework availability). Silence stdout/stderr
# temporarily during imports so the terminal stays clean.
with open(os.devnull, 'w') as _devnull:
    with redirect_stdout(_devnull), redirect_stderr(_devnull):
        from pdf_loader import load_pdf
        from embedding_creator import create_embeddings
        from qa_chain import create_qa_chain, LLM_MODEL_NAME, OLLAMA_API_BASE_URL


def main():
    # Cache frequently used functions
    path_exists = os.path.exists
    path_expanduser = os.path.expanduser
    
    # If we have a saved FAISS metadata file, use the stored PDF path and load the index
    meta_file = "faiss.db"
    pdf_path = None
    
    if path_exists(meta_file):
        try:
            import json  # Import here for faster startup when not needed
            with open(meta_file, "r") as f:
                meta = json.loads(f.read())
            stored = meta.get("source_path")
            if stored and path_exists(stored):
                pdf_path = stored
                print(f"[INFO] Using cached PDF: {pdf_path}")
        except Exception as e:
            print(f"[WARN] Could not load cached PDF path: {e}")

    while not pdf_path or not path_exists(path_expanduser(pdf_path)):
        pdf_path = input("Enter path of your PDF file: ").strip('" \'')

    docs = load_pdf(pdf_path)

    # Pass source_path so embedding_creator will save it in metadata when building index
    vector_store = create_embeddings(docs, source_path=pdf_path)
    qa_chain, _, _ = create_qa_chain(vector_store)

    # Print and verify Ollama HTTP API base URL and that the model is available locally.
    print(f"[INFO] Using Ollama HTTP API base URL: {OLLAMA_API_BASE_URL}")
    # Check HTTP reachability (best-effort). Use requests if available.
    try:
        import requests
        r = requests.get(OLLAMA_API_BASE_URL, timeout=3)
        print(f"[INFO] Ollama HTTP endpoint reachable (status {r.status_code})")
    except Exception as e:
        print(f"[WARN] Could not reach Ollama HTTP endpoint at {OLLAMA_API_BASE_URL}: {e}")

    # Check that the model is present in `ollama list` (best-effort). If `ollama` CLI isn't available
    # this will warn but not stop execution.
    try:
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
        out = (res.stdout or "") + (res.stderr or "")
        if LLM_MODEL_NAME in out:
            print(f"[INFO] Ollama model '{LLM_MODEL_NAME}' found locally.")
        else:
            print(f"[WARN] Ollama model '{LLM_MODEL_NAME}' not found in `ollama list` output. Run `ollama pull {LLM_MODEL_NAME}` if needed.")
    except Exception as e:
        print(f"[WARN] Could not run `ollama list` to verify models: {e}")
    
    print("\n[READY] Ask questions about the PDF content (type 'exit' to quit, 'help' for tips):")
    
    help_text = """
Tips for better answers:
- Be specific in your questions
- Ask about one topic at a time
- For definitions, include 'what is' or 'define'
- For processes, ask 'how to' or 'steps for'
- Type 'exit' to quit
"""
    
    while True:
        query = input("\nQuestion: ").strip()
        query_lower = query.lower()
        
        if query_lower == "exit":
            print("\nThank you for using the PDF QA system!")
            break
        if query_lower == "help":
            print(help_text)
            continue
        if not query:
            continue
            
        try:
            # Use the LCEL chain to get answer with timeout protection
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            with ThreadPoolExecutor() as executor:
                future = executor.submit(qa_chain.invoke, query)
                try:
                    answer = future.result(timeout=30)  # 30 second timeout
                except TimeoutError:
                    print("\n[ERROR] Question took too long to answer. Try asking a more specific question.\n")
                    continue
            
            # Clean up and format the answer
            answer = answer.strip()
            if not answer or answer.lower() == query_lower:
                print("\nAnswer: I don't know or couldn't find relevant information in the PDF.\n")
            else:
                # Truncate very long answers while preserving complete sentences
                if len(answer) > 1500:
                    answer = answer[:1500].rsplit(".", 1)[0] + "..."
                print(f"\nAnswer: {answer}\n")
                
        except Exception as e:
            print(f"\n[ERROR] Failed to get answer: {e}\n"
                  "Try rephrasing your question or asking about a different topic.\n")


if __name__ == "__main__":
    main()