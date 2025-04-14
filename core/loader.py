from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
import os


def filter_metadata(metadata):
    """Flatten or remove non-primitive metadata values."""
    return {
        k: v[0] if isinstance(v, list) and len(v) > 0 else v
        for k, v in metadata.items()
        if isinstance(v, (str, int, float, bool))
        or (
            isinstance(v, list)
            and len(v) > 0
            and isinstance(v[0], (str, int, float, bool))
        )
    }


def load_and_split_file(filepath):
    """Load and split a single file into chunks, while sanitizing metadata."""
    loader = UnstructuredLoader(filepath)
    raw_docs = loader.load()

    # Sanitize metadata
    clean_docs = [
        Document(page_content=doc.page_content, metadata=filter_metadata(doc.metadata))
        for doc in raw_docs
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(clean_docs)


def load_and_split_folder(folder_path, glob_pattern="**/*.md"):
    """
    Load and split documents from a folder.

    Args:
        folder_path (str): The path to the folder containing documents
        glob_pattern (str): Pattern to match files (default: "**/*.md" for all markdown files)

    Returns:
        List of Document objects with sanitized metadata and split into chunks
    """
    # Use DirectoryLoader to get all matching files
    loader = DirectoryLoader(folder_path, glob=glob_pattern, show_progress=True)

    # Get the raw documents
    raw_docs = loader.load()

    # Process and sanitize metadata for each document
    clean_docs = []
    for doc in raw_docs:
        # Make sure the source file is in the metadata
        if "source" not in doc.metadata and hasattr(doc, "metadata"):
            doc.metadata["source"] = os.path.basename(
                doc.metadata.get("source", "unknown")
            )

        # Filter the metadata
        clean_metadata = filter_metadata(doc.metadata)

        # Create a new document with clean metadata
        clean_docs.append(
            Document(page_content=doc.page_content, metadata=clean_metadata)
        )

    # Split the documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(clean_docs)
