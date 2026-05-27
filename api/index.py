import os
import re
from functools import lru_cache
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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

HOME_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MedAssist</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #152238;
      --muted: #607086;
      --line: #dbe3ef;
      --brand: #146c94;
      --brand-dark: #0f4f6b;
      --accent: #21a67a;
      --danger: #b42318;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #eef7fb 0%, var(--bg) 42%, #ffffff 100%);
    }

    .shell {
      width: min(1040px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 22px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .mark {
      display: grid;
      place-items: center;
      width: 44px;
      height: 44px;
      border-radius: 8px;
      color: #ffffff;
      background: var(--brand);
      font-size: 24px;
      font-weight: 700;
    }

    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.1;
    }

    .subtitle {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 14px;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 36px;
      padding: 0 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--brand-dark);
      background: rgba(255, 255, 255, 0.78);
      font-size: 14px;
      white-space: nowrap;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--accent);
    }

    main {
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 18px;
      align-items: start;
    }

    .chat,
    aside {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 18px 60px rgba(20, 108, 148, 0.08);
    }

    .messages {
      height: min(62vh, 620px);
      min-height: 420px;
      overflow-y: auto;
      padding: 18px;
    }

    .message {
      max-width: 86%;
      margin: 0 0 14px;
      padding: 12px 14px;
      border-radius: 8px;
      line-height: 1.5;
      white-space: pre-wrap;
    }

    .assistant {
      border: 1px solid var(--line);
      background: #f8fbff;
    }

    .user {
      margin-left: auto;
      color: #ffffff;
      background: var(--brand);
    }

    .composer {
      display: flex;
      gap: 10px;
      padding: 14px;
      border-top: 1px solid var(--line);
      background: #ffffff;
      border-radius: 0 0 8px 8px;
    }

    textarea {
      width: 100%;
      min-height: 48px;
      max-height: 150px;
      resize: vertical;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      font: inherit;
      color: var(--ink);
    }

    button {
      min-width: 104px;
      height: 48px;
      border: 0;
      border-radius: 8px;
      color: #ffffff;
      background: var(--brand);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    button:disabled {
      cursor: wait;
      background: #8aa6b6;
    }

    aside {
      padding: 18px;
    }

    aside h2 {
      margin: 0 0 12px;
      font-size: 18px;
    }

    .source {
      padding: 12px 0;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    .source strong {
      display: block;
      margin-bottom: 6px;
      color: var(--ink);
    }

    .notice {
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }

    .error {
      color: var(--danger);
    }

    @media (max-width: 840px) {
      header,
      main {
        display: block;
      }

      .status {
        margin-top: 14px;
      }

      aside {
        margin-top: 18px;
      }

      .messages {
        height: 58vh;
        min-height: 360px;
      }

      .message {
        max-width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand">
        <div class="mark">+</div>
        <div>
          <h1>MedAssist</h1>
          <p class="subtitle">Ask medical questions grounded in the uploaded reference PDF.</p>
        </div>
      </div>
      <div class="status"><span class="dot"></span> Live on Vercel</div>
    </header>

    <main>
      <section class="chat" aria-label="MedAssist chat">
        <div id="messages" class="messages">
          <div class="message assistant">Hello, I am MedAssist. Ask a medical question and I will answer from the available reference material.</div>
        </div>
        <form id="chat-form" class="composer">
          <textarea id="question" name="question" placeholder="Ask about symptoms, medicines, or conditions..." required></textarea>
          <button id="send" type="submit">Send</button>
        </form>
      </section>

      <aside>
        <h2>Sources</h2>
        <div id="sources">
          <p class="notice">Source snippets will appear here after each answer.</p>
        </div>
        <p class="notice">MedAssist is for educational use and is not a substitute for professional medical advice.</p>
      </aside>
    </main>
  </div>

  <script>
    const form = document.getElementById("chat-form");
    const input = document.getElementById("question");
    const messages = document.getElementById("messages");
    const sources = document.getElementById("sources");
    const send = document.getElementById("send");

    function addMessage(text, type) {
      const node = document.createElement("div");
      node.className = `message ${type}`;
      node.textContent = text;
      messages.appendChild(node);
      messages.scrollTop = messages.scrollHeight;
      return node;
    }

    function renderSources(items) {
      sources.innerHTML = "";
      if (!items.length) {
        sources.innerHTML = '<p class="notice">No source snippets returned.</p>';
        return;
      }

      items.forEach((item) => {
        const node = document.createElement("div");
        node.className = "source";
        const title = document.createElement("strong");
        title.textContent = `Page ${item.page}`;
        const content = document.createElement("span");
        content.textContent = item.content;
        node.append(title, content);
        sources.appendChild(node);
      });
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = input.value.trim();
      if (!question) return;

      addMessage(question, "user");
      input.value = "";
      send.disabled = true;
      const pending = addMessage("Thinking...", "assistant");

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ question }),
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "The server could not answer.");
        }

        pending.textContent = data.answer;
        pending.classList.remove("error");
        renderSources(data.sources || []);
      } catch (error) {
        pending.textContent = error.message;
        pending.classList.add("error");
      } finally {
        send.disabled = false;
        input.focus();
      }
    });
  </script>
</body>
</html>"""


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


@app.get("/", response_class=HTMLResponse)
def root():
    return HOME_HTML


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
