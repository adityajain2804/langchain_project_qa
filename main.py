# main.py
import warnings

# Suppress a known LangChain/Pydantic compatibility user warning on Python 3.14+
# This is safe to suppress locally; consider updating langchain_core or Python version later.
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.*")

from pdf_loader import load_pdf
from embedding_creator import create_embeddings
from qa_chain import create_qa_chain


def main():
    pdf_path = input("Enter path of your PDF file: ")
    docs = load_pdf(pdf_path)
    
    vector_store = create_embeddings(docs)
    qa, retriever, llm = create_qa_chain(vector_store)
    
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