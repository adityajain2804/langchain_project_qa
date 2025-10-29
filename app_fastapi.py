# app_fastapi.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import answer_question

app = FastAPI(title="PDF QA API (FastAPI Version)", version="1.0")

# ✅ Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later e.g. ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🏠 Root route
@app.get("/")
async def home():
    return {"message": "✅ FastAPI PDF QA backend is running!"}


# 🧾 Request body schema
class QuestionRequest(BaseModel):
    question: str


# 🤖 Main QA endpoint
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        return {"error": "Please provide a valid question."}
    
    answer = answer_question(question)
    return {"answer": answer}
