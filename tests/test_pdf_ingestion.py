"""PDF add, list, and delete against local Chroma with fake MinerU output and figure text."""

import json
from unittest.mock import Mock
from uuid import uuid4

import chromadb
import pytest
from llama_index.core import Settings, load_index_from_storage
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.chroma import ChromaVectorStore

import pdf_ingestion
import services
import storage
from retrieval import load_index_nodes


def describe(image_path, caption):
    if image_path.name == "broken.jpg":
        raise RuntimeError("model unavailable")
    return "员工提交申请，经理审批后提交财务。"


@pytest.fixture
def mineru_output(tmp_path):
    folder = tmp_path / "mineru"
    (folder / "images").mkdir(parents=True)
    table = "<table><tr><td>级别</td><td>上限</td></tr><tr><td>L1</td><td>500</td></tr></table>"
    items = [
        {"type": "text", "text": "报销制度", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "员工出差后十个工作日内提交报销。", "page_idx": 0},
        {"type": "image", "img_path": "images/flow.jpg", "image_caption": ["图 1"], "page_idx": 1},
        {"type": "image", "img_path": "images/broken.jpg", "image_caption": [], "page_idx": 1},
        {"type": "table", "table_caption": ["表 1"], "table_body": table, "page_idx": 2},
        {"type": "footer", "text": "第 3 页", "page_idx": 2},
    ]
    path = folder / "policy_content_list.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def knowledge_base():
    """An empty knowledge base whose vector store keeps text, like the Chroma server."""
    Settings.embed_model = MockEmbedding(embed_dim=8)
    collection = chromadb.EphemeralClient().get_or_create_collection(f"kb_{uuid4().hex}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    context = services.build_storage_context(
        SimpleDocumentStore(), SimpleIndexStore(), vector_store
    )
    return context, collection


def write_pdf(folder, name, content):
    path = folder / name
    path.write_bytes(content)
    return path


def ingest(pdf, knowledge_base, content_list, tmp_path):
    context, collection = knowledge_base
    return pdf_ingestion.ingest_pdf(
        pdf,
        context,
        collection,
        work_dir=tmp_path / "work",
        parse=lambda *_: content_list,
        describe=describe,
    )


def test_failed_figures_are_skipped_and_reported(mineru_output):
    skipped = []
    blocks = pdf_ingestion.content_to_blocks(mineru_output, describe, skipped)
    assert [kind for kind, _, _ in blocks] == ["text", "figure", "table"]
    assert [page for _, _, page in blocks] == [1, 2, 3]
    assert skipped == [{"page": 2, "caption": "", "reason": "RuntimeError: model unavailable"}]
    assert all("第 3 页" not in text for _, text, _ in blocks)


def test_add_list_and_delete_keep_all_three_stores_consistent(
    mineru_output, knowledge_base, tmp_path
):
    context, collection = knowledge_base
    first = write_pdf(tmp_path, "报销制度.pdf", b"%PDF first")
    second = write_pdf(tmp_path, "leave.pdf", b"%PDF second")

    first_id, first_count, skipped = ingest(first, knowledge_base, mineru_output, tmp_path)
    second_id, second_count, _ = ingest(second, knowledge_base, mineru_output, tmp_path)

    assert len(skipped) == 1
    index = load_index_from_storage(context, index_id=storage.INDEX_ID)
    nodes = load_index_nodes(index)
    assert len(nodes) == first_count + second_count == collection.count()
    assert {node.metadata["file_name"] for node in nodes} == {"报销制度.pdf", "leave.pdf"}
    assert {node.metadata["content_type"] for node in nodes} == {"text", "figure", "table"}
    documents = {row["doc_id"]: row for row in pdf_ingestion.list_pdfs(context.docstore)}
    assert documents[first_id]["file_name"] == "报销制度.pdf"
    assert documents[first_id]["chunks"] == first_count
    assert documents[first_id]["skipped_figures"] == 1

    pdf_ingestion.delete_pdf(context, first_id)

    index = load_index_from_storage(context, index_id=storage.INDEX_ID)
    nodes = load_index_nodes(index)
    assert {node.ref_doc_id for node in nodes} == {second_id}
    assert collection.count() == second_count
    assert [row["doc_id"] for row in pdf_ingestion.list_pdfs(context.docstore)] == [second_id]
    # A deleted PDF can be added again.
    assert ingest(first, knowledge_base, mineru_output, tmp_path)[0] == first_id


def test_duplicate_pdf_is_rejected_before_parsing(mineru_output, knowledge_base, tmp_path):
    context, collection = knowledge_base
    pdf = write_pdf(tmp_path, "policy.pdf", b"%PDF same")
    ingest(pdf, knowledge_base, mineru_output, tmp_path)
    parse = Mock()
    with pytest.raises(ValueError, match="already in the knowledge base"):
        pdf_ingestion.ingest_pdf(pdf, context, collection, work_dir=tmp_path, parse=parse)
    parse.assert_not_called()


def test_pdf_without_content_writes_nothing(knowledge_base, tmp_path):
    context, collection = knowledge_base
    content_list = tmp_path / "empty_content_list.json"
    content_list.write_text(json.dumps([{"type": "footer", "text": "1", "page_idx": 0}]))
    with pytest.raises(ValueError, match="Nothing was added"):
        ingest(
            write_pdf(tmp_path, "blank.pdf", b"%PDF blank"), knowledge_base, content_list, tmp_path
        )
    assert collection.count() == 0
    assert not context.index_store.get_index_struct(storage.INDEX_ID)


def test_only_pdf_files_are_accepted(knowledge_base, tmp_path):
    with pytest.raises(ValueError, match="Only PDF"):
        ingest(write_pdf(tmp_path, "notes.txt", b"text"), knowledge_base, None, tmp_path)


def test_deleting_an_unknown_document_is_rejected(knowledge_base):
    with pytest.raises(ValueError, match="not in the knowledge base"):
        pdf_ingestion.delete_pdf(knowledge_base[0], "pdf-missing")
