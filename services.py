"""Model configuration, indexing, scoped retrieval, and persistent chat memory."""

from llama_index.core import Settings, StorageContext
from llama_index.core.base.response.schema import Response, StreamingResponse
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode

from prompts import QA_PROMPT
from retrieval import build_metadata_filters, build_policy_retriever

EMPTY_ANSWER_MESSAGE = (
    "The language model returned no answer. Check the DashScope API key, "
    "regional endpoint, and account quota, then retry."
)


def validated_answer_stream(chunks):
    """Preserve streaming while rejecting a provider's empty output."""
    has_content = False
    for chunk in chunks:
        has_content = has_content or bool(chunk.strip())
        yield chunk
    if not has_content:
        raise ValueError(EMPTY_ANSWER_MESSAGE)


class PolicyQueryEngine(RetrieverQueryEngine):
    """Reject empty answers before display or persistence in CLI chat history."""

    def _query(self, query_bundle):
        response = super()._query(query_bundle)
        # The pinned DashScope adapter can turn provider errors into empty output.
        if isinstance(response, StreamingResponse):
            response.response_gen = validated_answer_stream(response.response_gen)
        elif isinstance(response, Response) and not (response.response or "").strip():
            raise ValueError(EMPTY_ANSWER_MESSAGE)
        return response


def configure_settings(llm, embed_model):
    """Set process-wide models without constructing or invoking providers."""
    Settings.llm = llm
    Settings.embed_model = embed_model


def split_documents(documents):
    """Split into 512-token nodes with 50-token overlap and inherited metadata."""
    return SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(documents)


def build_storage_context(docstore, index_store, vector_store):
    """Keep node text, index structure, and embeddings in their configured stores."""
    return StorageContext.from_defaults(
        docstore=docstore,
        index_store=index_store,
        vector_store=vector_store,
    )


def build_query_engine(index, llm, *, advanced=False, reranker=None, category=None):
    """Create a streaming engine with optional scope for either retrieval mode.

    The CLI and web interface use advanced retrieval. The vector-only mode is
    retained for baseline comparisons and lightweight service tests.
    """
    filters = build_metadata_filters(category)
    if advanced:
        if reranker is None:
            raise ValueError("Advanced retrieval requires a reranker.")
        retriever = build_policy_retriever(index, llm, category=category)
        postprocessors = [reranker]
    else:
        retriever = index.as_retriever(similarity_top_k=3, filters=filters)
        postprocessors = []

    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=ResponseMode.COMPACT,
        streaming=True,
        text_qa_template=QA_PROMPT,
    )
    return PolicyQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=postprocessors,
    )


def build_chat_memory(chat_store, session_key):
    """Restore one chat history with a 3,000-token context budget."""
    return ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=session_key,
        token_limit=3000,
    )
