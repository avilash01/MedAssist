# MedAssist RAG

MedAssist RAG is a local medical question-answering chatbot built with Chainlit, LangChain, FAISS, HuggingFace embeddings, and a local GGUF Llama-style model. It answers user questions using retrieved context from uploaded medical PDF documents and shows source snippets used for the answer.

## Features

- Medical document question answering
- PDF ingestion with `PyPDFLoader`
- Chunking with `RecursiveCharacterTextSplitter`
- Local FAISS vector database
- HuggingFace sentence-transformer embeddings
- Local LLM inference using `CTransformers`
- Chainlit chat interface
- Source document snippets included in responses
- Guardrail prompt that avoids making up answers when context is missing

## Tech Stack

- Python
- Chainlit
- LangChain
- FAISS
- HuggingFace Embeddings
- CTransformers
- TinyLlama GGUF model
- PDF document retrieval

## Project Structure

```text
medical_bot/
├── data/
│   ├── ingest.py                # Builds the FAISS vector DB from PDFs
│   ├── model.py                 # Chainlit RAG chatbot
│   ├── Medicine.pdf             # Example medical source document
│   ├── requirements.txt
│   ├── models/                  # Local GGUF model folder
│   └── vectorstores/db_faiss/   # Generated FAISS index
├── .chainlit/
├── .gitignore
└── README.md
```

## How It Works

1. Add medical PDFs to the `data/` folder.
2. Run `ingest.py` to load PDFs, split them into chunks, embed them, and store them in FAISS.
3. Run the Chainlit app from `model.py`.
4. Ask medical questions in the chat UI.
5. The app retrieves the most relevant chunks and passes them to the local LLM.
6. The answer is returned with source snippets.

## Setup

```bash
cd data
python -m venv medical_bot_env
medical_bot_env\Scripts\activate
pip install -r requirements.txt
```

## Model Setup

Place the local GGUF model here:

```text
data/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

The model path is configured in `data/model.py`.

## Build the Vector Database

```bash
cd data
python ingest.py
```

This creates:

```text
data/vectorstores/db_faiss/
```

## Run the Chatbot

```bash
cd data
chainlit run model.py
```

Then open the Chainlit URL shown in the terminal.

## Main Files

- `data/ingest.py` - loads PDFs, creates chunks, builds FAISS index
- `data/model.py` - loads FAISS, local LLM, and Chainlit chat flow
- `data/requirements.txt` - Python dependencies

## Important Note

This project is for educational and portfolio use. It is not a replacement for professional medical advice, diagnosis, or treatment.
