from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def build_vector_store(docs, persist_directory="./chroma_db"):
    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(
        docs=docs, embeddings=embeddings, persist_directory=persist_directory
    )
    return db
