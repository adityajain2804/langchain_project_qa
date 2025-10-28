# pdf_loader.py
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import os


def load_pdf(pdf_path: str):
    """Load a PDF and return a list of LangChain Document objects.

    Accepts absolute or relative Windows paths and expands ~. Raises FileNotFoundError
    with a clear message if the file is not found.
    """
    pdf_path = str(pdf_path).strip()
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

    loader = PyPDFLoader(str(p))
    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} document chunks from {p}")
    return documents