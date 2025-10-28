# pdf_loader.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pathlib import Path
import os


def split_text(text, chunk_size=500, overlap=50):
    """Split text into smaller chunks with optimized overlap."""
    # Pre-process text to normalize whitespace and improve splitting
    text = ' '.join(text.split())  # Normalize whitespace
    
    # Extract potential acronyms and definitions for special handling
    import re
    acronyms = re.findall(r'\b([A-Z]{2,})\b[\s]*[-–—:]+[\s]*([^.!?\n]+[.!?\n])', text)
    definitions = {}
    for acr, defn in acronyms:
        definitions[acr] = defn.strip()
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to find a good breaking point
        if end < text_len:
            # Prioritize breaking at paragraph/sentence boundaries
            for sep in [". ", ".\n", "! ", "? ", "\n\n", "\n", " "]:
                pos = text.rfind(sep, start, end + overlap)
                if pos != -1:
                    end = pos + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            # Add relevant definitions to chunks containing acronyms
            chunk_acronyms = re.findall(r'\b([A-Z]{2,})\b', chunk)
            relevant_defs = []
            for acr in chunk_acronyms:
                if acr in definitions:
                    relevant_defs.append(f"{acr}: {definitions[acr]}")
            
            if relevant_defs:
                chunk = chunk + "\n\nDefinitions:\n" + "\n".join(relevant_defs)
            
            chunks.append(chunk)
        
        # Smaller overlap for efficiency but ensure complete sentences
        start = max(end - overlap, start + 1)
    
    return chunks

def load_pdf(pdf_path: str) -> list[Document]:
    """Load and chunk a PDF file with optimized settings."""
    pdf_path = str(pdf_path).strip()
    # Remove surrounding quotes if the path was pasted with quotes
    if (pdf_path.startswith('"') and pdf_path.endswith('"')) or (
        pdf_path.startswith("'") and pdf_path.endswith("'")):
        pdf_path = pdf_path[1:-1]
    # Try common normalizations
    p = Path(pdf_path).expanduser()
    if not p.exists():
        # try absolute path resolution
        p_abs = Path(os.path.abspath(pdf_path))
        if p_abs.exists():
            p = p_abs
        else:
            # handle file:// URI
            if pdf_path.lower().startswith("file://"):
                candidate = Path(pdf_path[7:])
                if candidate.exists():
                    p = candidate
                else:
                    raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            else:
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Load the PDF
    loader = PyPDFLoader(str(p))
    documents = loader.load()
    
    # Split each page into smaller chunks
    chunked_docs = []
    for doc in documents:
        chunks = split_text(doc.page_content)
        page_num = doc.metadata.get("page", 0)
        
        # Create new documents for each chunk
        for i, chunk in enumerate(chunks):
            chunked_docs.append(Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk": i,
                    "page": page_num
                }
            ))
    
    print(f"[INFO] Loaded and split PDF into {len(chunked_docs)} optimized chunks from {p}")
    return chunked_docs