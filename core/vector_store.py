from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def build_vector_store(docs, persist_directory="./chroma_db", rebuilt_db=False):
    db = None
    embeddings = HuggingFaceEmbeddings()

    if rebuilt_db:
        db = Chroma.from_documents(
            docs=docs,
            embeddings=embeddings,
            persist_directory=persist_directory,
        )
    else:
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )

    return db
