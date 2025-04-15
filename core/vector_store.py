# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def build_vector_store(docs, persist_directory="./chroma_db", rebuilt_db=False):
    db = None
    embedding = OpenAIEmbeddings()

    if rebuilt_db:
        db = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=persist_directory,
        )
    else:
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )

    return db
