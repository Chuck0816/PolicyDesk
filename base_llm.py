"""Initialize the Qwen language model and local HuggingFace embedding model."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def create_models():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Set DASHSCOPE_API_KEY in .env before building or asking questions.")

    # Delayed imports keep preview and unit tests free of model downloads.
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.dashscope import DashScope

    llm = DashScope(model_name="qwen-plus", api_key=api_key)
    embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    )
    return llm, embed_model
