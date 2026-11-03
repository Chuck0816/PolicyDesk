"""Unit and database integration tests for indexing, retrieval, and conversation state."""

import os
from uuid import uuid4

import pytest
from llama_index.core import Document, Settings, VectorStoreIndex, load_index_from_storage
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import ChatMessage, MockLLM
from llama_index.core.schema import NodeRelationship, TextNode
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore

import services
import storage
from ingestion import load_documents
from main import build_index
from prompts import CHAT_PROMPT, QA_PROMPT, RICH_PROMPT


@pytest.fixture
def functions():
    return services


@pytest.fixture(autouse=True)
def test_models():
    # These are test doubles only, never an application provider option.
    Settings.llm = MockLLM(max_tokens=16)
    Settings.embed_model = MockEmbedding(embed_dim=8)


def test_settings_keep_the_supplied_models(functions):
    llm, embed = MockLLM(), MockEmbedding(embed_dim=8)
    functions.configure_settings(llm, embed)
    assert Settings.llm is llm
    assert Settings.embed_model is embed


def test_splitting_preserves_source_and_links(functions):
    document = Document(text="Managers approve leave. " * 400, metadata={"file_name": "policy.txt"})
    nodes = functions.split_documents([document])
    assert len(nodes) > 1
    ids = {node.node_id for node in nodes}
    for node in nodes:
        assert node.metadata["file_name"] == "policy.txt"
        assert node.ref_doc_id == document.doc_id
        for relation in (NodeRelationship.NEXT, NodeRelationship.PREVIOUS):
            if relation in node.relationships:
                assert node.relationships[relation].node_id in ids


def test_storage_context_keeps_all_three_roles(functions):
    # Small in-memory doubles isolate this unit test; the application uses databases.
    docstore, index_store, vector_store = (
        SimpleDocumentStore(),
        SimpleIndexStore(),
        SimpleVectorStore(),
    )
    context = functions.build_storage_context(docstore, index_store, vector_store)
    assert context.docstore is docstore
    assert context.index_store is index_store
    assert context.vector_store is vector_store


def test_query_engine_streams_and_returns_sources(functions):
    index = VectorStoreIndex(
        [TextNode(text="Managers approve leave.", metadata={"file_name": "leave.txt"})]
    )
    response = functions.build_query_engine(index, Settings.llm).query("Who approves leave?")
    assert "".join(response.response_gen)
    assert response.source_nodes[0].metadata["file_name"] == "leave.txt"


def test_memory_restores_the_matching_session_only(functions):
    store = SimpleChatStore()
    memory = functions.build_chat_memory(store, "review")
    memory.put(ChatMessage(role="user", content="Who approves leave?"))
    memory.put(ChatMessage(role="assistant", content="Managers."))
    assert len(functions.build_chat_memory(store, "review").get()) == 2
    assert functions.build_chat_memory(store, "another-session").get() == []


def test_loading_includes_two_files_and_a_manual_document():
    assert {doc.metadata["file_name"] for doc in load_documents()} == {
        "travel_policy.txt",
        "leave_policy.txt",
        "manual_faq",
    }


def test_three_prompt_styles_render_the_same_inputs():
    for prompt in (QA_PROMPT, CHAT_PROMPT, RICH_PROMPT):
        text = prompt.format(context_str="Only this policy", query_str="Who approves?")
        assert "Only this policy" in text and "Who approves?" in text


@pytest.mark.skipif(
    os.getenv("POLICYDESK_INTEGRATION") != "1", reason="Requires the database services."
)
def test_database_build_reload_and_chat_restore(monkeypatch):
    monkeypatch.setattr(storage, "KB_NAME", f"policydesk_test_{uuid4().hex[:10]}")
    context, collection = storage.connect_storage(services, create=True)
    build_index(context, collection, services)
    with pytest.raises(ValueError, match="already"):
        build_index(context, collection, services)
    reconnected, _ = storage.connect_storage(services)
    index = load_index_from_storage(reconnected, index_id=storage.INDEX_ID)
    engine = services.build_query_engine(index, Settings.llm)
    response = engine.query("Who approves leave?")
    assert "".join(response.response_gen) and response.source_nodes
    key = f"{storage.KB_NAME}:chat:review"
    memory = services.build_chat_memory(storage.connect_chat_store(), key)
    chat = CondenseQuestionChatEngine.from_defaults(
        query_engine=engine, llm=Settings.llm, memory=memory
    )
    assert chat.chat("Who approves leave?").source_nodes
    restored = services.build_chat_memory(storage.connect_chat_store(), key)
    assert len(restored.get()) == 2
    assert services.build_chat_memory(storage.connect_chat_store(), key + ":other").get() == []
