from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def build_vector_store(docs):
    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    return db
