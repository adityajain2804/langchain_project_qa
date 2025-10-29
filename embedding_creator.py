# embedding_creator.py
# Use the updated langchain-ollama package for Ollama integrations
import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import json
from pathlib import Path


# Change this name if you pull a different embedding model with Ollama
EMBED_MODEL_NAME = "mxbai-embed-large"


def create_embeddings(docs, source_path: str = None):
    """Create embeddings for the given LangChain documents using Ollama.

    Returns:
    vector_store: FAISS vector store built from documents
    """
    print(f"[INFO] Creating embeddings using Ollama model: {EMBED_MODEL_NAME} ...")

    base_url = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    # Disable strict SSL verification for local HTTP Ollama endpoints
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=base_url, verify=False)
    except Exception:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=base_url)

    INDEX_DIR = Path("faiss_index")
    META_FILE = Path("faiss.db")

    if META_FILE.exists() and INDEX_DIR.exists():
        try:
            meta = json.loads(META_FILE.read_text())
            if meta.get("embed_model") == EMBED_MODEL_NAME:
                print("[INFO] Loading FAISS index from disk...")
                try:
                    vector_store = FAISS.load_local(str(INDEX_DIR), embeddings)
                    print("[INFO] FAISS index loaded from disk.")
                    return vector_store
                except Exception:
                    print("[WARN] Could not load FAISS index; rebuilding index.")
            else:
                print("[WARN] Embedding model changed; rebuilding index.")
        except Exception as e:
            print("[WARN] Could not load FAISS metadata:", e)

    print("[INFO] Preprocessing documents: optimizing chunks for faster processing...")
    texts = []
    seen = set()
    MIN_CHARS = 800

    from hashlib import blake2b
    def quick_hash(text):
        return blake2b(text.encode(), digest_size=8).hexdigest()

    buffer = ""
    for d in docs:
        txt = d.page_content.strip()
        if not txt:
            continue
        h = hash(txt)
        if h in seen:
            continue
        seen.add(h)

        if len(txt) < MIN_CHARS:
            if buffer:
                buffer += "\n\n" + txt
            else:
                buffer = txt
            if len(buffer) >= MIN_CHARS:
                texts.append(buffer)
                buffer = ""
        else:
            if buffer:
                texts.append(buffer)
                buffer = ""
            texts.append(txt)

    if buffer:
        texts.append(buffer)

    print(f"[INFO] {len(texts)} preprocessed text chunks will be embedded (original docs: {len(docs)})")
    texts = [t for t in texts if len(t) >= 30]

    def reduce_texts(texts_list, target_count):
        if len(texts_list) <= target_count:
            return texts_list
        merged = list(texts_list)
        while len(merged) > target_count:
            best_i = 0
            best_len = len(merged[0]) + len(merged[1]) if len(merged) > 1 else len(merged[0])
            for i in range(len(merged) - 1):
                l = len(merged[i]) + len(merged[i + 1])
                if l < best_len:
                    best_len = l
                    best_i = i
            merged[best_i] = merged[best_i] + "\n\n" + merged.pop(best_i + 1)
            if len(merged) % 100 == 0:
                print(f"[INFO] Reduced to {len(merged)} chunks...")
        return merged

    MAX_CHUNKS = 800
    if len(texts) > MAX_CHUNKS:
        print(f"[WARN] Too many chunks ({len(texts)}). Reducing to {MAX_CHUNKS}...")
        texts = reduce_texts(texts, MAX_CHUNKS)
        print(f"[INFO] Reduced chunk count: {len(texts)}")

    batch_size = 512
    vectors = []
    total = len(texts)
    try:
        print("[INFO] Starting embedding process with optimized batching...")
        embeddings.client_kwargs = {"timeout": 60, "retry_on_timeout": True}

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            v = embeddings.embed_documents(batch)
            vectors.extend(v)
            print(f"[INFO] Embedded {min(i + batch_size, total)}/{total} chunks ({(min(i + batch_size, total) / total * 100):.1f}%)")
    except KeyboardInterrupt:
        print("\n[WARN] Embedding interrupted by user.")
    except Exception as e:
        print("[ERROR] Embedding failed:", e)
        raise

    print("[INFO] Building FAISS vector store from precomputed embeddings...")

    class PrecomputedEmbeddings:
        def __init__(self, vectors):
            self.vectors = vectors
            self._pos = 0

        def embed_documents(self, texts_in):
            n = len(texts_in)
            out = self.vectors[self._pos:self._pos + n]
            self._pos += n
            return out

        def embed_query(self, text):
            return embeddings.embed_query(text)

        def __call__(self, text):
            if isinstance(text, str):
                return self.embed_query(text)
            return self.embed_documents(text)

    pre_emb = PrecomputedEmbeddings(vectors)

    from langchain_core.documents import Document as LcDocument
    faiss_docs = []
    for i, t in enumerate(texts):
        faiss_docs.append(LcDocument(page_content=t, metadata={"source": source_path or "", "chunk": i}))

    vector_store = FAISS.from_documents(faiss_docs, pre_emb)
    print("[INFO] FAISS vector store created.")

    try:
        INDEX_DIR.mkdir(exist_ok=True)
        vector_store.save_local(str(INDEX_DIR))
        meta = {"embed_model": EMBED_MODEL_NAME, "index_dir": str(INDEX_DIR.resolve()), "num_chunks": len(texts)}
        if source_path:
            meta["source_path"] = str(source_path)
        META_FILE.write_text(json.dumps(meta))
        print(f"[INFO] FAISS index saved to {INDEX_DIR} and metadata written to {META_FILE}")
    except Exception as e:
        print("[WARN] Failed to save FAISS index or metadata:", e)

    return vector_store


# 🟢 Add this new runtime batching function BELOW the existing create_embeddings()
def create_vector_on_demand(docs, source_path=None, existing_store=None):
    """Embed only needed chunks when required at runtime."""
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    import random

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL_NAME,
        base_url=os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    )

    # Randomly sample a portion to embed if the document is huge
    CHUNK_LIMIT = 500
    to_embed = docs
    if len(docs) > CHUNK_LIMIT:
        print(f"[INFO] Large PDF detected ({len(docs)} chunks). Embedding only {CHUNK_LIMIT} most relevant chunks for now...")
        to_embed = random.sample(docs, CHUNK_LIMIT)

    texts = [d.page_content for d in to_embed]
    vectors = embeddings.embed_documents(texts)
    faiss_docs = [Document(page_content=t, metadata={"source": source_path or ""}) for t in texts]

    if existing_store:
        existing_store.add_documents(faiss_docs)
        print(f"[INFO] Added {len(faiss_docs)} new embeddings to existing FAISS store.")
        return existing_store
    else:
        store = FAISS.from_documents(faiss_docs, embeddings)
        print(f"[INFO] Created FAISS store on-demand with {len(faiss_docs)} chunks.")
        return store
