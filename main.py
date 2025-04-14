import os
import time
from rich.console import Console
from rich.markdown import Markdown
from typing import Dict, Any, List
from dotenv import load_dotenv
from core.loader import load_and_split_folder
from core.vector_store import build_vector_store
from core.llm import get_vertex_llm
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import pickle
import hashlib
import os.path


def calculate_docs_hash(docs: List) -> str:
    """Calculate a hash of the document collection to detect changes"""
    content_hash = hashlib.md5()
    for doc in docs:
        content_hash.update(doc.page_content.encode())
    return content_hash.hexdigest()


def create_personal_ai_helper(
    folder_path: str,
    glob_pattern: str = "**/*.md",
    vector_store_path: str = "vector_store.pkl",
) -> Dict[str, Any]:
    """
    Creates a personal AI helper that acts as a therapist, teacher, and friend.

    Args:
        folder_path: Path to the folder containing personal notes
        glob_pattern: File pattern to match (default: all markdown files)
        vector_store_path: Path to save/load the vector store

    Returns:
        Dictionary containing the QA chain and metadata
    """
    console = Console()
    console.print("Loading and processing documents...", style="yellow")

    # Fix path escaping if present
    cleaned_path = folder_path.replace("\\", "")

    # Load and process documents
    docs = load_and_split_folder(
        folder_path=cleaned_path,
        glob_pattern=glob_pattern,
    )
    console.print(f"Loaded {len(docs)} document chunks", style="green")

    # Calculate hash of current documents
    docs_hash = calculate_docs_hash(docs)

    # Check if we have a saved vector store and if it matches our current documents
    db = None
    vector_store_exists = os.path.exists(vector_store_path)

    if vector_store_exists:
        try:
            console.print(
                "Found existing vector store, checking if it's current...",
                style="yellow",
            )
            # Load the hash from the companion file
            hash_file = f"{vector_store_path}.hash"
            if os.path.exists(hash_file):
                with open(hash_file, "r") as f:
                    saved_hash = f.read().strip()

                if saved_hash == docs_hash:
                    console.print("Loading existing vector store...", style="green")
                    with open(vector_store_path, "rb") as f:
                        db = pickle.load(f)
                    console.print("Vector store loaded from cache", style="green")
                else:
                    console.print(
                        "Documents have changed, rebuilding vector store...",
                        style="yellow",
                    )
            else:
                console.print(
                    "No hash file found, rebuilding vector store...", style="yellow"
                )
        except Exception as e:
            console.print(f"Error loading vector store: {e}", style="red")
            console.print("Will rebuild vector store from scratch", style="yellow")

    # If no valid vector store was loaded, build a new one
    if db is None:
        console.print("Building new vector store...", style="yellow")
        db = build_vector_store(docs)

        # Save the vector store and its hash
        with open(vector_store_path, "wb") as f:
            pickle.dump(db, f)

        with open(f"{vector_store_path}.hash", "w") as f:
            f.write(docs_hash)

        console.print("Vector store built and saved", style="green")

    # Set up retriever with reasonable defaults
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # Get LLM
    console.print("Setting up language model...", style="yellow")
    llm = get_vertex_llm()
    console.print("Language model ready", style="green")

    # Create a comprehensive prompt template for personal helper interaction
    prompt = ChatPromptTemplate.from_template(
        """
    You are a personal AI helper named Emre's Assistant, carefully crafted to provide support based on the user's personal notes and documents.
    Think of yourself as a blend of a therapist, wise teacher, and trusted friend.

    Your primary qualities:
    1. Therapeutic: You listen carefully, show empathy, and help explore thoughts and feelings
    2. Wise Teacher: You provide thoughtful guidance based on context and knowledge
    3. Honest Friend: You give brutally honest feedback when needed, but always with compassion
    4. Trustworthy: You protect private information and maintain confidentiality

    When responding:
    - ALWAYS provide a substantive response based on the context
    - If you don't find relevant information in the context, be honest but helpful
    - Refer to insights and patterns found in the user's personal notes when possible
    - Be warm and conversational, but direct and honest
    - Provide personalized advice and observations about the user's life, challenges, and growth
    - Avoid being judgmental or making assumptions without evidence

    Here's the relevant context from the user's personal notes:
    {context}

    User's question or request:
    {input}

    Provide a thoughtful, personalized response. NEVER return an empty response.
    """
    )

    # Create the document chain for processing retrieved documents
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # Create the retrieval chain combining retrieval and document processing
    qa_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=document_chain
    )

    return {"qa_chain": qa_chain, "doc_count": len(docs), "notes_path": cleaned_path}


def interactive_session(helper_system: Dict[str, Any]):
    """
    Run an interactive session with the personal AI helper.

    Args:
        helper_system: Dictionary containing the QA chain and metadata
    """
    console = Console()
    qa_chain = helper_system["qa_chain"]

    console.print(
        f"\n[bold green]Personal AI Helper initialized with {helper_system['doc_count']} document chunks[/bold green]"
    )
    console.print(
        "[bold blue]Your notes at: [/bold blue]" + helper_system["notes_path"]
    )
    console.print("\n[bold]Welcome to your personal AI helper session.[/bold]")
    console.print(
        "This AI has been trained on your personal notes and is here to help you."
    )
    console.print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        user_input = input("\n[You]: ")

        if user_input.lower() in ["exit", "quit"]:
            console.print("\n[bold]Thank you for the conversation. Take care![/bold]\n")
            break

        try:
            # Process the user's input through the QA chain
            console.print("\n[AI Helper]: ", end="")

            # Show a spinner or some indication that processing is happening
            console.print("Thinking...", style="dim")

            # Process the query
            response = qa_chain.invoke({"input": user_input})

            # Clear the "thinking" message
            console.print("\r" + " " * 20 + "\r", end="")

            # Ensure we have a non-empty response
            answer = response.get("answer", "").strip()
            if not answer:
                answer = "I couldn't find specific information about that in your notes, but I'm here to help. Could you provide more details or ask another question?"

            # Display the AI's response
            console.print(Markdown(answer))

        except Exception as e:
            console.print(
                f"\n[bold red]Error: Something went wrong - {str(e)}[/bold red]"
            )
            console.print("Let's try again with a different question.")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Get notes path from environment variable or use default
    notes_path = os.getenv(
        "NOTES_PATH", os.path.expanduser("~/Desktop/Emre's All Notes/Google Keep")
    )
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "vector_store.pkl")

    # Create the personal AI helper system
    helper = create_personal_ai_helper(
        folder_path=notes_path,
        glob_pattern=os.getenv("NOTES_GLOB", "**/*.md"),
        vector_store_path=vector_store_path,
    )

    # Set environment variable to avoid tokenizer parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Start the interactive session
    interactive_session(helper)
