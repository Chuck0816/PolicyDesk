"""Core services for model configuration, policy indexing, retrieval, and chat memory."""

from llama_index.core import Settings, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode

from prompts import QA_PROMPT


def configure_settings(llm, embed_model):
    """Configure the default language and embedding models for the application.

    Args:
        llm: Initialized language model used for policy-answer generation.
        embed_model: Initialized embedding model used for indexing and retrieval.

    Updates the process-wide LlamaIndex Settings without creating or calling models.
    """
    Settings.llm = llm
    Settings.embed_model = embed_model


def split_documents(documents):
    """Split policy documents into overlapping, metadata-aware text nodes.

    Args:
        documents: Source Document objects loaded from the policy corpus.

    Returns:
        Nodes with inherited source metadata and document relationships. Chunks
        use a 512-token budget with a target overlap of 50 tokens.
    """
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes


def build_storage_context(docstore, index_store, vector_store):
    """Assemble the application's database-backed storage components.

    Args:
        docstore: Store for complete node text, metadata, and relationships.
        index_store: Store for the index structure required to reload the index.
        vector_store: Store for node embeddings used in similarity retrieval.

    Returns:
        A StorageContext using the supplied store instances.
    """
    # Reuse the externally configured stores without creating new connections.

    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        index_store=index_store,
        vector_store=vector_store,
    )
    return storage_context


def build_query_engine(index, llm):
    """Create a streaming policy query engine with retrieval and answer synthesis.

    Args:
        index: Loaded vector index containing the company's policy nodes.
        llm: Language model used to generate answers from retrieved context.

    Returns:
        A RetrieverQueryEngine that retrieves up to three relevant nodes and
        synthesizes a streamed answer with the compact response mode and QA_PROMPT.
    """
    # Retrieve the most relevant policy nodes.
    retriever = index.as_retriever(similarity_top_k=3)

    # Generate a streamed answer grounded in the retrieved policy context.
    synthesizer = get_response_synthesizer(
        llm=llm, response_mode=ResponseMode.COMPACT, streaming=True, text_qa_template=QA_PROMPT
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
    )


def build_chat_memory(chat_store, session_key):
    """Create a token-limited conversation buffer backed by the supplied chat store.

    Args:
        chat_store: Persistent message store shared by conversation sessions.
        session_key: Identifier selecting the conversation's existing history.

    Returns:
        A ChatMemoryBuffer with a 3,000-token context budget. Reusing the key
        restores history; the budget does not delete messages from the store.
    """

    memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=session_key,
        token_limit=3000,
    )

    return memory
