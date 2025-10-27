# embedding_creator.py
# Use the updated langchain-ollama package for Ollama integrations
import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# Change this name if you pull a different embedding model with Ollama
EMBED_MODEL_NAME = "mxbai-embed-large"




def create_embeddings(docs):
    """Create embeddings for the given LangChain documents using Ollama.


    Returns:
    vector_store: FAISS vector store built from documents
    """
    print(f"[INFO] Creating embeddings using Ollama model: {EMBED_MODEL_NAME} ...")


    # If you're running Ollama's HTTP API on localhost, set OLLAMA_API_BASE_URL to
    # e.g. "http://localhost:11434" in your environment so embeddings use the same host.
    base_url = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=base_url)


    # Optionally compute embeddings first (useful for printing / debugging)
    try:
        texts = [d.page_content for d in docs]
        vectors = embeddings.embed_documents(texts)
        print("[INFO] Sample embeddings (first 2 vectors, first 10 dims):")
        for i, v in enumerate(vectors[:2]):
            print(f"Vector {i+1}: {v[:10]} ...")
    except Exception as e:
        print("[WARN] Could not call embed_documents for debug print:", e)


    # Build FAISS vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    print("[INFO] FAISS vector store created.")


    return vector_store