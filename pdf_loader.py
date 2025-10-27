# pdf_loader.py
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str):
    """Load a PDF and return a list of LangChain Document objects."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} document chunks from {pdf_path}")
    return documents