import os
import time
import shutil
from rich.console import Console
from rich.markdown import Markdown
from typing import Dict, Any, List
from dotenv import load_dotenv
from core.loader import load_and_split_folder  # Assuming core.loader exists
from core.vector_store import build_vector_store  # Assuming core.vector_store exists
from core.llm import get_vertex_llm  # Assuming core.llm exists
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import hashlib
import json
from datetime import datetime

# --- Configuration ---
HISTORY_BASE_DIR = "history"  # Base directory for conversation logs
YOUR_NAME = "Emre"  # Your name for personalization


def calculate_docs_hash(docs: List[Document]) -> str:
    """Calculate a hash of the document collection to detect changes"""
    content_hash = hashlib.md5()
    sorted_docs = sorted(
        docs, key=lambda d: (d.metadata.get("source", ""), d.page_content)
    )
    for doc in sorted_docs:
        content_hash.update(doc.page_content.encode("utf-8"))
        metadata_str = json.dumps(doc.metadata, sort_keys=True)
        content_hash.update(metadata_str.encode("utf-8"))
    return content_hash.hexdigest()


def create_personal_ai_helper(
    folder_path: str,
    glob_pattern: str = "**/*.md",
    persist_directory: str = "./chroma_db",
) -> Dict[str, Any]:
    """
    Creates a personal AI helper using personal notes.

    Args:
        folder_path: Path to the folder containing personal notes
        glob_pattern: File pattern to match (default: all markdown files)
        persist_directory: Directory to persist the Chroma vector store

    Returns:
        Dictionary containing the QA chain and metadata
    """
    console = Console()
    console.print("Loading and processing documents...", style="yellow")

    cleaned_path = folder_path.replace("\\", "")

    docs = load_and_split_folder(
        folder_path=cleaned_path,
        glob_pattern=glob_pattern,
    )
    if not docs:
        console.print(
            f"[bold red]Error: No documents found matching pattern '{glob_pattern}' in '{cleaned_path}'. Exiting.[/bold red]"
        )
        exit(1)
    console.print(f"Loaded {len(docs)} document chunks", style="green")

    docs_hash = calculate_docs_hash(docs)
    hash_file = os.path.join(persist_directory, "docs_hash.json")

    rebuild_db = True
    if os.path.exists(persist_directory) and os.path.exists(hash_file):
        try:
            console.print(
                "Found existing Chroma DB, checking if it's current...", style="yellow"
            )
            with open(hash_file, "r") as f:
                hash_data = json.load(f)
                saved_hash = hash_data.get("hash", "")

            if saved_hash == docs_hash:
                console.print(
                    "Documents unchanged, using existing Chroma DB", style="green"
                )
                rebuild_db = False
            else:
                console.print(
                    "Documents have changed, rebuilding Chroma DB...", style="yellow"
                )
                # Ensure directory removal before rebuilding
                if os.path.exists(persist_directory):
                    shutil.rmtree(persist_directory, ignore_errors=True)
        except Exception as e:
            console.print(f"Error checking vector store hash: {e}", style="red")
            console.print("Will rebuild Chroma DB from scratch", style="yellow")
            if os.path.exists(persist_directory):
                shutil.rmtree(persist_directory, ignore_errors=True)

    # Ensure directory exists before building/loading
    os.makedirs(persist_directory, exist_ok=True)

    if rebuild_db:
        console.print("Building new Chroma vector store...", style="yellow")
        db = build_vector_store(
            docs, persist_directory=persist_directory, rebuilt_db=True
        )
        with open(hash_file, "w") as f:
            json.dump({"hash": docs_hash, "timestamp": time.time()}, f)
        console.print("Chroma DB built and saved", style="green")
    else:
        db = build_vector_store(
            docs, persist_directory=persist_directory, rebuilt_db=False
        )
        console.print("Loaded existing Chroma DB", style="green")

    # Retriever Setup
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 15,
            "fetch_k": 50,
            # "lambda_mult": 0.6
        },
    )

    console.print("Setting up language model...", style="yellow")
    llm = get_vertex_llm()
    console.print("Language model ready", style="green")

    prompt = ChatPromptTemplate.from_template(
        f"""
You are "{YOUR_NAME}'s Assistant", an AI companion designed for supportive and insightful conversations based *strictly* on the provided context from {YOUR_NAME}'s personal notes. Your persona blends empathy, wisdom, and friendly honesty.

**Your Core Role & Tone:**
*   **Empathetic Listener:** Actively listen, validate feelings expressed in the input and context. Use phrases like "It sounds like...", "I hear that you're feeling...", "That seems understandable given...".
*   **Insightful Guide:** Gently guide reflection by asking open-ended questions related to the context. Connect current input to patterns or themes found in the notes.
*   **Conversational Partner:** Maintain a natural, warm, and flowing conversational style. Avoid overly robotic or formal language. Use "I" statements where appropriate (e.g., "Based on the notes, I see a pattern of...").
*   **Grounded in Context:** **Your primary knowledge source is the provided context below.** Base your answers, reflections, and questions directly on this information. If the context doesn't contain relevant information, state that clearly (e.g., "The provided notes don't seem to cover [topic]. Could you share more?") rather than speculating or using general knowledge outside of basic conversational pleasantries.
*   **Honest but Kind:** If the notes suggest challenging patterns or contradictions, reflect them back gently and constructively.
*   **Safety:** Do NOT provide medical diagnoses or advice. Do NOT act as a replacement for a licensed therapist. If the user expresses severe distress or intent for self-harm, gently suggest seeking professional help from a qualified human.

**Output Format (CRITICAL):**
*   **Always respond in Markdown format.**
*   **Structure for Obsidian:**
    *   Use standard Markdown (headings `##`, `###`, lists `*`, `-`, bold `**text**`, italics `*text*`).
    *   **Incorporate relevant hashtags (#tag)** within your response text to categorize key themes, feelings, topics, or insights. Examples: `#reflection`, `#feeling/anxiety`, `#goal/project-x`, `#pattern/procrastination`, `#insight/connection`, `#person/name`, `#topic/work`. Be thoughtful, relevant, and consistent with tagging.
    *   Use headings (`##`, `###`) appropriately for structure within longer responses.

**Interaction Flow:**
1.  Analyze the user's input: `{{input}}`.
2.  Carefully review the relevant context retrieved from {YOUR_NAME}'s notes, provided below.
3.  Synthesize information *only* from the context that relates to the input.
4.  Formulate a response that fulfills the persona and formatting requirements above.
5.  Ask clarifying or reflective questions grounded in the context if appropriate.

---
**Context from {YOUR_NAME}'s Notes:**
{{context}}
---

**User's Input:**
{{input}}

**Your Markdown Response:**
"""
    )

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    qa_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=document_chain
    )

    # Return the chain and useful metadata
    return {
        "qa_chain": qa_chain,
        "doc_count": len(docs),
        "notes_path": cleaned_path,
        "vector_db_path": persist_directory,
    }


def save_conversation_turn(session_dir: str, user_input: str, ai_response: str):
    """Saves a single turn of the conversation to a Markdown file."""
    try:
        timestamp = datetime.now()
        file_name = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.md"
        file_path = os.path.join(session_dir, file_name)

        frontmatter = f"""---
date: {timestamp.isoformat()}
tags:
  - ai-conversation
  - therapy-session
  - session/{os.path.basename(session_dir)}
---

"""
        # Use YOUR_NAME variable here
        content = f"{frontmatter}## User Query\n\n{user_input}\n\n## {YOUR_NAME}'s Assistant Response\n\n{ai_response}\n"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        Console().print(
            f"[bold red]Error saving conversation turn to {file_path}: {e}[/bold red]"
        )


def interactive_session(helper_system: Dict[str, Any]):
    """Run an interactive session, saving conversations to structured Markdown files."""
    console = Console()
    qa_chain = helper_system["qa_chain"]

    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(HISTORY_BASE_DIR, session_timestamp)
    try:
        os.makedirs(session_dir, exist_ok=True)
        console.print(f"[dim]Saving conversation history to: {session_dir}[/dim]")
    except Exception as e:
        console.print(
            f"[bold red]Error creating history directory {session_dir}: {e}. History will not be saved.[/bold red]"
        )
        session_dir = None

    console.print(
        f"\n[bold green]Personal AI Helper initialized ({helper_system['doc_count']} chunks)[/bold green]"
    )
    console.print(f"[bold blue]Notes Path:[/bold blue] {helper_system['notes_path']}")
    console.print(
        f"[bold blue]Vector DB:[/bold blue] {helper_system['vector_db_path']}"
    )
    # Use YOUR_NAME variable here
    console.print(f"\n[bold]Welcome back, {YOUR_NAME}! Let's talk.[/bold]")
    console.print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = console.input(
                f"\n[bold cyan][{YOUR_NAME}]: [/bold cyan]"
            )  # Use YOUR_NAME for input prompt
        except EOFError:
            console.print("\n[bold]Exiting session.[/bold]\n")
            break

        if user_input.lower() in ["exit", "quit"]:
            console.print("\n[bold]It was good talking. Take care![/bold]\n")
            break
        if not user_input.strip():
            continue

        try:
            with console.status(
                "[bold yellow]Assistant is thinking...[/bold yellow]", spinner="dots"
            ):
                response = qa_chain.invoke({"input": user_input})

            answer = response.get("answer", "").strip()
            if not answer:
                answer = "# Note\n\nI couldn't formulate a specific response based on the provided notes for that query. Could we explore it differently, or perhaps focus on another topic?"
            elif not answer.startswith("#"):
                answer = f"# Response\n\n{answer}"

            # Use YOUR_NAME variable here
            console.print(f"\n[bold magenta][{YOUR_NAME}'s Assistant]:[/bold magenta]")
            console.print(Markdown(answer))

            if session_dir:
                save_conversation_turn(session_dir, user_input, answer)

        except Exception as e:
            console.print(f"\n[bold red]An error occurred: {str(e)}[/bold red]")
            console.print("Sorry about that. Let's try again.")


if __name__ == "__main__":
    load_dotenv()

    # Get configuration from environment variables or use defaults
    notes_path = os.getenv("NOTES_PATH", os.path.expanduser("~/Documents/MyNotes"))
    chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    notes_glob = os.getenv("NOTES_GLOB", "**/*.md")
    history_path_config = os.getenv("HISTORY_PATH", HISTORY_BASE_DIR)
    HISTORY_BASE_DIR = history_path_config  # Update global variable

    os.makedirs(HISTORY_BASE_DIR, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    console = Console()
    console.print("[bold]Initializing Personal AI Helper...[/bold]")

    try:
        # Make sure YOUR_NAME is accessible if needed within create_personal_ai_helper
        # (In this case, it's used directly in the prompt string defined inside)
        helper = create_personal_ai_helper(
            folder_path=notes_path,
            glob_pattern=notes_glob,
            persist_directory=chroma_db_path,
        )
        interactive_session(helper)
    except Exception as e:
        console.print(
            f"[bold red]Fatal Error during Initialization or Session: {e}[/bold red]"
        )
        exit(1)
