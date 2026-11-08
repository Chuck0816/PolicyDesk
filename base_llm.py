"""Create the Qwen provider and local embedding and relevance models."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def create_models():
    """Initialize models lazily so previews and offline tests need no weights."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Set DASHSCOPE_API_KEY in .env before building or asking questions.")

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.dashscope import DashScope

    llm = DashScope(model_name="qwen-plus", api_key=api_key)
    embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    )
    return llm, embed_model


def create_reranker():
    """Create the top-four BGE reranker; callers reuse it across questions."""
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

    # Full precision supports local CPU execution without an FP16 requirement.
    return FlagEmbeddingReranker(
        model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large"),
        top_n=4,
        use_fp16=False,
    )
