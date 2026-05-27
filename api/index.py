import os
import re
from functools import lru_cache
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from groq import Groq
from pypdf import PdfReader
from pydantic import BaseModel, Field


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(ROOT_DIR, "data", "Medicine.pdf")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

SYSTEM_PROMPT = """You are MedAssist, a careful medical information assistant.
Use only the provided context to answer. If the context does not contain the
answer, say you do not know. Do not provide a diagnosis, and advise users to
consult a qualified medical professional for urgent or personal medical advice."""


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class Source(BaseModel):
    content: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


app = FastAPI(title="MedAssist API")


@lru_cache(maxsize=1)
def get_chunks() -> List[Tuple[str, int]]:
    reader = PdfReader(PDF_PATH)
    chunks: List[Tuple[str, int]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if not text:
            continue
        start = 0
        while start < len(text):
            chunk = text[start : start + CHUNK_SIZE]
            chunks.append((chunk, page_number))
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not configured")
    return Groq()


def tokenize(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-zA-Z]{3,}", text.lower())}


def retrieve_context(question: str, limit: int = 4) -> List[Tuple[str, int]]:
    query_terms = tokenize(question)
    if not query_terms:
        return get_chunks()[:limit]

    scored = []
    for chunk, page in get_chunks():
        chunk_terms = tokenize(chunk)
        score = len(query_terms & chunk_terms)
        if score:
            scored.append((score, chunk, page))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(chunk, page) for _, chunk, page in scored[:limit]] or get_chunks()[:limit]


@app.get("/")
def root():
    return {
        "name": "MedAssist API",
        "status": "ok",
        "endpoints": ["/api/health", "/api/chat"],
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        docs = retrieve_context(payload.question)
        context = "\n\n".join(
            f"Source page {page}:\n{chunk}" for chunk, page in docs
        )
        response = get_groq_client().chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.2,
            max_tokens=512,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question:\n{payload.question}\n\n"
                        "Answer:"
                    ),
                },
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = response.choices[0].message.content or "I do not know."
    sources = [
        Source(content=chunk[:500], page=page)
        for chunk, page in docs
    ]
    return ChatResponse(answer=answer, sources=sources)
