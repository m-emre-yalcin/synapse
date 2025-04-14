from langchain_google_vertexai import VertexAI
from core.env import init_env
import os


def get_vertex_llm():
    init_env()

    llm = VertexAI(
        model_name="gemini-2.5-pro-preview-03-25",
        project=os.getenv("PROJECT_ID"),
        location=os.getenv("REGION"),
        max_output_tokens=2048,
        temperature=0.8,
    )
    return llm
