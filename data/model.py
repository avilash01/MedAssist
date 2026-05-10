import os
import chainlit as cl
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH = os.path.join(
    BASE_DIR,
    "vectorstores",
    "db_faiss"
)
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)

custom_prompt_template = """
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know.
Don't try to make up an answer.
Context:
{context}
Question:
{question}
Give only the helpful answer below:
"""
def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

def load_llm():
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        config={
            "max_new_tokens": 128,
            "temperature": 0.5,
            "context_length": 1024
           
        }
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_kwargs={"k": 2}
        ),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt
        }
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(
        llm,
        qa_prompt,
        db
    )
    return qa
@cl.on_chat_start
async def start():
    chain = qa_bot()
    cl.user_session.set(
        "chain",
        chain
    )
    await cl.Message(
        content="Medical Bot is Ready! Ask your medical questions."
    ).send()
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    print("User Question:", message.content)
    try:
        response = await chain.ainvoke({
            "query": message.content
        })
        answer = response["result"]
        source_docs = response.get("source_documents")
        if source_docs:
            answer += "\n\nSources Used:\n"
            for i, doc in enumerate(source_docs):
                answer += f"\nSource {i+1}:\n"
                answer += doc.page_content[:300]
                answer += "\n"
    except Exception as e:
        answer = f"Error: {str(e)}"
    await cl.Message(
        content=answer
    ).send()