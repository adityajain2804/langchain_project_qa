# qa_chain.py
# Use the updated Ollama LLM from langchain-ollama
import os
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA


# Change this to the LLM you pulled with ollama
# Use the exact model name you pulled. Updated per your note.
LLM_MODEL_NAME = "gpt-oss:120b-cloud"

# Base URL for the local Ollama HTTP API. You can override by setting the
# OLLAMA_API_BASE_URL environment variable, e.g.:
#   $env:OLLAMA_API_BASE_URL = "http://localhost:11434"
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")




def create_qa_chain(vector_store):
    """Create a RetrievalQA chain using an Ollama LLM and the provided vector store."""
    print(f"[INFO] Creating QA chain with LLM: {LLM_MODEL_NAME}")
    # Instantiate the Ollama LLM via the langchain-ollama adapter
    # Set a low temperature for more deterministic answers
    # Pass base_url so the adapter uses the local Ollama HTTP API
    llm = OllamaLLM(model=LLM_MODEL_NAME, temperature=0.2, base_url=OLLAMA_API_BASE_URL)


    # Use a slightly larger k to retrieve more context for the question
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Use the 'refine' chain type which often produces better, non-echoing answers
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)
    # Return both the QA chain and the retriever so callers can inspect retrieved docs if needed
    return qa, retriever, llm