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
    # Disable strict SSL verification for local HTTP Ollama endpoints to avoid
    # unnecessary SSL context delays on local installs. If you use HTTPS with
    # valid certs, remove verify=False.
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=base_url, verify=False)
    except Exception:
        # Some versions of the Ollama client may not accept `verify` or may raise
        # a Pydantic ValidationError. Fall back to the default constructor.
        embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=base_url)

    # Persisted FAISS index location and metadata
    INDEX_DIR = Path("faiss_index")
    META_FILE = Path("faiss.db")

    # Fast path: load existing index if metadata matches
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
                    # fall through to rebuilding index
                    print("[WARN] Could not load FAISS index from disk; rebuilding index.")
            else:
                print("[WARN] Embedding model changed since index was saved; rebuilding index.")
        except Exception as e:
            print("[WARN] Could not load FAISS metadata, rebuilding index:", e)

    # Aggressive preprocessing to minimize chunks while preserving quality
    print("[INFO] Preprocessing documents: optimizing chunks for faster processing...")
    texts = []
    seen = set()
    # Higher threshold for merging to reduce total chunks
    MIN_CHARS = 800  # Increased minimum size for better chunking efficiency
    
    # Pre-compute hashes for faster deduplication
    from hashlib import blake2b
    def quick_hash(text):
        return blake2b(text.encode(), digest_size=8).hexdigest()

    # Merge tiny consecutive chunks into larger ones to reduce total count
    buffer = ""
    for d in docs:
        txt = d.page_content.strip()
        if not txt:
            continue
        # simple dedupe by exact text hash
        h = hash(txt)
        if h in seen:
            continue
        seen.add(h)

        if len(txt) < MIN_CHARS:
            # accumulate into buffer
            if buffer:
                buffer += "\n\n" + txt
            else:
                buffer = txt
            # if buffer is big enough, flush
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

    # Remove very small texts
    texts = [t for t in texts if len(t) >= 30]

    # If there are an excessive number of chunks, reduce them by merging adjacent
    # chunks until we reach a practical target. This keeps retrieval quality
    # reasonable while drastically reducing embedding time on very large PDFs.
    def reduce_texts(texts_list, target_count):
        if len(texts_list) <= target_count:
            return texts_list
        merged = list(texts_list)
        # repeatedly merge the shortest adjacent pairs until under target
        import heapq

        # compute lengths
        while len(merged) > target_count:
            # merge adjacent pairs greedily: find smallest combined length
            best_i = 0
            best_len = len(merged[0]) + len(merged[1]) if len(merged) > 1 else len(merged[0])
            for i in range(len(merged) - 1):
                l = len(merged[i]) + len(merged[i+1])
                if l < best_len:
                    best_len = l
                    best_i = i
            # merge best_i and best_i+1
            merged[best_i] = merged[best_i] + "\n\n" + merged.pop(best_i+1)
            # stop if too slow
            if len(merged) % 100 == 0:
                # lightweight progress hint
                print(f"[INFO] Reduced to {len(merged)} chunks...")
        return merged

    # Reduced maximum chunks for better performance while maintaining quality
    MAX_CHUNKS = 800  # Reduced from 1200 for faster processing
    if len(texts) > MAX_CHUNKS:
        print(f"[WARN] Too many chunks ({len(texts)}). Reducing to {MAX_CHUNKS} for faster processing...")
        texts = reduce_texts(texts, MAX_CHUNKS)
        print(f"[INFO] Reduced chunk count: {len(texts)} for optimal performance")

    # Embed in batches with optimized settings
    # Use larger batch size and request pipelining for faster embedding
    batch_size = 512  # Increased batch size for fewer API calls
    vectors = []
    total = len(texts)
    try:
        print("[INFO] Starting embedding process with optimized batching...")
        
        # Configure embedding parameters for speed
        embeddings.client_kwargs = {
            "timeout": 60,  # Increased timeout for larger batches
            "retry_on_timeout": True
        }
        
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            v = embeddings.embed_documents(batch)
            vectors.extend(v)
            print(f"[INFO] Embedded {min(i+batch_size, total)}/{total} chunks ({(min(i+batch_size, total)/total*100):.1f}%)")
    except KeyboardInterrupt:
        print("\n[WARN] Embedding interrupted by user. Building partial index from already-computed vectors...")
    except Exception as e:
        print("[ERROR] Embedding failed:", e)
        raise

    # Build FAISS index from precomputed vectors without re-embedding
    print("[INFO] Building FAISS vector store from precomputed embeddings...")

    # Create a tiny wrapper that hands back precomputed embeddings in order
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
            # For query-time embedding, use the actual Ollama embedder
            return embeddings.embed_query(text)
            
        def __call__(self, text):
            # Support callable interface for single text embedding
            if isinstance(text, str):
                return self.embed_query(text)
            return self.embed_documents(text)

    pre_emb = PrecomputedEmbeddings(vectors)

    # Recreate Document objects aligned with the preprocessed texts so FAISS can attach metadata
    from langchain_core.documents import Document as LcDocument
    faiss_docs = []
    for i, t in enumerate(texts):
        faiss_docs.append(LcDocument(page_content=t, metadata={"source": source_path or "", "chunk": i}))

    vector_store = FAISS.from_documents(faiss_docs, pre_emb)
    print("[INFO] FAISS vector store created.")

    # Save index and metadata for future runs
    try:
        INDEX_DIR.mkdir(exist_ok=True)
        vector_store.save_local(str(INDEX_DIR))
        meta = {
            "embed_model": EMBED_MODEL_NAME,
            "index_dir": str(INDEX_DIR.resolve()),
            "num_chunks": len(texts),
        }
        if source_path:
            meta["source_path"] = str(source_path)
        META_FILE.write_text(json.dumps(meta))
        print(f"[INFO] FAISS index saved to {INDEX_DIR} and metadata written to {META_FILE}")
    except Exception as e:
        print("[WARN] Failed to save FAISS index or metadata:", e)

    return vector_store