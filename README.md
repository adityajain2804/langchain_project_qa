# LangChain + Ollama PDF Q&A

A fast and efficient PDF question-answering system using LangChain and Ollama.

## 🚀 Features

- Optimized PDF processing with smart chunking
- Fast embedding creation with batched processing
- Persistent FAISS index for quick reloads
- Intelligent retrieval for accurate answers
- Special handling of acronyms and definitions
- Minimal dependencies for better performance

## � Requirements

- Python 3.9+
- Ollama running locally (default: http://localhost:11434)
- 4GB+ RAM recommended

## 🛠️ Setup Instructions

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate      # Windows
   source venv/bin/activate     # Linux/Mac
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## 💡 Usage

1. Run the application:
   ```bash
   python main.py
   ```
2. Enter the path to your PDF file
3. Ask questions about the PDF content
4. Type 'exit' to quit

## 🔄 Performance Notes

- First run will create and save embeddings
- Subsequent runs with same PDF use cached embeddings
- FAISS index automatically persists for faster reloading
