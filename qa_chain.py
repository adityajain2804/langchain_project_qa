# qa_chain.py
# Use the updated Ollama LLM from langchain-ollama
import os
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA


LLM_MODEL_NAME = "gpt-oss:120b-cloud"

#   $env:OLLAMA_API_BASE_URL = "http://localhost:11434"
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")




def create_qa_chain(vector_store):
    """Create a RetrievalQA chain using an Ollama LLM and the provided vector store."""
    print(f"[INFO] Creating QA chain with LLM: {LLM_MODEL_NAME}")
    llm = OllamaLLM(model=LLM_MODEL_NAME, temperature=0.2, base_url=OLLAMA_API_BASE_URL)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)
    return qa, retriever, llm