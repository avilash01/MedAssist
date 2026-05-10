import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = BASE_DIR
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstores", "db_faiss")


def create_vector_db():
    print("📂 Loading PDF files...")

    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    if not documents:
        print("PDF files not found. Please add PDFs to the folder.")
        return

    print(f"Loaded {len(documents)} documents")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    print(f"Created {len(texts)} chunks")

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("Building FAISS vector database...")
    db = FAISS.from_documents(texts, embeddings)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)

    print(f"✅ Vector DB saved at: {DB_FAISS_PATH}")


if __name__ == "__main__":
    create_vector_db()