# qa_chain.py
import random
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

LLM_MODEL_NAME = "gpt-oss:120b-cloud"
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")

# Optimize prompt for concise, accurate answers
QA_PROMPT = PromptTemplate(
    template="""You are a precise and helpful assistant. Answer questions using only the provided context. Be concise and direct.

Context: {context}

Question: {question}

Rules:
- If the answer isn't in the context, say "I don't know"
- For definitions, give a short, precise answer
- For explanations, give a brief, clear paragraph
- Don't include irrelevant information
- Don't use external knowledge

Answer: """,
    input_variables=["context", "question"]
)




def create_qa_chain(vector_store):
    """Create a RetrievalQA chain using an Ollama LLM and the provided vector store."""
    print(f"[INFO] Creating QA chain with LLM: {LLM_MODEL_NAME}")
    llm = OllamaLLM(
        model=LLM_MODEL_NAME,
        temperature=random.uniform(0.7, 1.2),  # 🌀 Har run me thoda random creativity
        top_p=random.uniform(0.8, 1.0),        # 🧠 Sampling randomness aur badha
        repeat_penalty=1.1,                    # 🔁 Repeat kam kare
        base_url=OLLAMA_API_BASE_URL,
        context_window=4096,
        num_ctx=4096
    )

    # Optimize retrieval settings for better answer accuracy
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",  # Changed to threshold-based retrieval
        search_kwargs={
            "k": 15,  # Increased for better context coverage
            "score_threshold": 0.2,  # Lowered threshold to catch more potential matches
            "fetch_k": 8,  # Fetch more candidates before filtering
        }
    )

        # Create a more efficient chain using LCEL
    rag_chain = (
        RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )
        | QA_PROMPT 
        | llm 
        | StrOutputParser()
    )
    return rag_chain, retriever, llm