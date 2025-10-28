# main.py
import warnings
import os
import subprocess

# Suppress a known LangChain/Pydantic compatibility user warning on Python 3.14+
# This is safe to suppress locally; consider updating langchain_core or Python version later.
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.*")

from pdf_loader import load_pdf
from embedding_creator import create_embeddings
from qa_chain import create_qa_chain, LLM_MODEL_NAME, OLLAMA_API_BASE_URL


def main():
    pdf_path = input("Enter path of your PDF file: ")
    docs = load_pdf(pdf_path)
    
    vector_store = create_embeddings(docs)
    qa, _retriever, llm = create_qa_chain(vector_store)

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
    
    print("\n[READY] Ask your question related to PDF content (type 'exit' to quit):")
    while True:
        query = input("\nQuestion: ")
        if query.strip().lower() == "exit":
            break
        try:
            # Fetch top retrieved documents (no debug prints) so we can build context
            docs_for_query = vector_store.similarity_search(query, k=4)

            # Construct a prompt explicitly using the retrieved context and call the LLM directly.
            context_text = "\n\n".join([d.page_content for d in docs_for_query])
            prompt = (
                "You are an assistant that answers questions using ONLY the provided context. "
                "If the answer is not contained in the context, reply with 'I don't know.'\n\n"
                f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer concisely:")

            # Try calling the underlying LLM directly. If that fails, fall back to qa.invoke.
            try:
                llm_result = llm.invoke({"input": prompt})
                if isinstance(llm_result, dict):
                    answer = llm_result.get("output_text") or llm_result.get("answer") or next(iter(llm_result.values()), "")
                else:
                    answer = llm_result
            except Exception:
                # Fallback: use the QA chain invocation
                result = qa.invoke({"query": query})
                if isinstance(result, dict):
                    answer = result.get("output_text") or result.get("answer") or next(iter(result.values()), "")
                else:
                    answer = result

            # If the model simply echoed the question or returned empty, use the top retrieved doc as the answer.
            if not answer or answer.strip().lower() == query.strip().lower():
                if docs_for_query:
                    # Provide a concise excerpt from the top document as the answer
                    answer = docs_for_query[0].page_content.strip()
                    # Truncate to a reasonable length
                    if len(answer) > 1500:
                        answer = answer[:1500].rsplit(" ", 1)[0] + "..."
                else:
                    answer = "I don't know."

            print(f"\nAnswer: {answer}\n")
        except Exception as e:
            print(f"\n[ERROR] Failed to get answer from chain: {e}\n")


if __name__ == "__main__":
    main()