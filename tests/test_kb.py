"""Knowledge-base operations: routing by file type and domain, listing, and deleting."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from llama_index.core.storage.docstore.types import RefDocInfo

import kb


@pytest.fixture
def oncall(monkeypatch):
    monkeypatch.setattr(kb.storage, "KB_DOMAIN", "oncall")


@pytest.mark.parametrize(
    "path, text, explicit, expected",
    [
        ("x.md", "", "runbook", "runbook"),
        ("x.md", "---\ndoc_type: postmortem\n---\n# A", None, "postmortem"),
        ("docs/runbooks/cart.md", "# Cart", None, "runbook"),
        ("Postmortems/2026/INC-1.md", "# INC", None, "postmortem"),
    ],
)
def test_markdown_type_comes_from_option_front_matter_or_folder(path, text, explicit, expected):
    assert kb.markdown_doc_type(path, text, explicit) == expected


def test_unknown_markdown_type_is_explained():
    with pytest.raises(ValueError, match="doc_type: runbook"):
        kb.markdown_doc_type("notes/cart.md", "# Cart")


def test_pdfs_go_through_pdf_ingestion(monkeypatch):
    ingest = Mock(return_value=("pdf-1", 3, [{"page": 2}]))
    monkeypatch.setattr(kb.pdf_ingestion, "ingest_pdf", ingest)
    parse = object()
    assert kb.add_document("a.PDF", "ctx", "col", work_dir="w", parse=parse) == (
        "pdf-1",
        3,
        {"skipped": [{"page": 2}]},
    )
    assert ingest.call_args.kwargs["parse"] is parse


def test_markdown_needs_the_oncall_domain(monkeypatch, tmp_path):
    monkeypatch.setattr(kb.storage, "KB_DOMAIN", "policy")
    (tmp_path / "a.md").write_text("# A", encoding="utf-8")
    with pytest.raises(ValueError, match="KB_DOMAIN=oncall"):
        kb.add_document(tmp_path / "a.md", "ctx", "col", work_dir="w")
    with pytest.raises(ValueError, match="Only PDF and Markdown"):
        kb.add_document(tmp_path / "a.docx", "ctx", "col", work_dir="w")


def test_postmortems_need_an_imported_catalog(oncall, tmp_path):
    path = tmp_path / "INC-1.md"
    path.write_text("---\ndoc_type: postmortem\nincident_id: INC-1\n---\n# A\nb", encoding="utf-8")
    with pytest.raises(ValueError, match="Import the service catalog"):
        kb.add_document(path, "ctx", "col", work_dir="w", graph_store=object())


def test_markdown_is_passed_to_graph_ingestion(oncall, monkeypatch, tmp_path):
    ingest = Mock(return_value=("md-1", 2, {"chunks": 2}))
    monkeypatch.setattr(kb.graph_ingestion, "ingest_markdown", ingest)
    path = tmp_path / "runbooks" / "cart.md"
    path.parent.mkdir()
    path.write_text("# Cart", encoding="utf-8")
    store, catalog = object(), object()
    kb.add_document(
        path,
        "ctx",
        "col",
        work_dir="w",
        graph_store=store,
        catalog=catalog,
        extraction=("extractor", "cache"),
    )
    kwargs = ingest.call_args.kwargs
    assert kwargs["doc_type"] == "runbook" and kwargs["graph_store"] is store
    assert (kwargs["extractor"], kwargs["cache"]) == ("extractor", "cache")


def info(node_count, **metadata):
    return RefDocInfo(node_ids=[f"n{i}" for i in range(node_count)], metadata=metadata)


def test_list_groups_catalog_documents_into_one_row():
    docstore = Mock()
    docstore.get_all_ref_doc_info.return_value = {
        "catalog-cart": info(1, doc_type="service_catalog", uploaded_at="2026-11-02"),
        "catalog-payment": info(2, doc_type="service_catalog", uploaded_at="2026-11-03"),
        "pdf-1": info(4, doc_type="uploaded_document", file_name="a.pdf", uploaded_at="2026-11-01"),
        "md-1": info(
            2,
            doc_type="postmortem",
            file_name="INC-1.md",
            uploaded_at="2026-11-05",
            graph_relations=6,
        ),
        "seed": info(1, doc_type="something_else"),
    }
    rows = kb.list_documents(docstore)
    assert [row["doc_id"] for row in rows] == ["catalog", "md-1", "pdf-1"]
    assert rows[0] | {"file_name": None} == {
        "doc_id": "catalog",
        "file_name": None,
        "doc_type": "service_catalog",
        "uploaded_at": "2026-11-03",
        "chunks": 3,
        "services": 2,
        "skipped_figures": 0,
        "graph_relations": 0,
    }
    assert rows[1]["graph_relations"] == 6 and rows[2]["doc_type"] == "pdf"


def test_delete_cleans_the_graph_then_the_vector_stores(monkeypatch):
    order = []
    monkeypatch.setattr(
        kb.graph_ingestion, "delete_document_graph", lambda s, d: order.append(("graph", d))
    )
    monkeypatch.setattr(kb.pdf_ingestion, "delete_pdf", lambda c, d: order.append(("vector", d)))
    context = SimpleNamespace(docstore=Mock())
    kb.delete_document(context, "md-1", graph_store=object())
    assert order == [("graph", "md-1"), ("vector", "md-1")]
    order.clear()
    kb.delete_document(context, "pdf-1")
    assert order == [("vector", "pdf-1")]


def test_delete_refuses_the_catalog_and_unknown_documents():
    context = SimpleNamespace(docstore=Mock())
    with pytest.raises(ValueError, match="cannot be deleted"):
        kb.delete_document(context, "catalog")
    context.docstore.ref_doc_exists.return_value = False
    with pytest.raises(ValueError, match="not in the knowledge base"):
        kb.delete_document(context, "pdf-9")


def test_import_catalog_needs_neo4j():
    with pytest.raises(ValueError, match="needs Neo4j"):
        kb.import_catalog("services: []", "", "ctx", "col", None)


def test_connect_graph_only_in_the_oncall_domain(monkeypatch):
    connect = Mock()
    monkeypatch.setattr(kb.storage, "connect_graph_store", connect)
    monkeypatch.setattr(kb.storage, "KB_DOMAIN", "policy")
    assert kb.connect_graph() == (None, None)
    connect.assert_not_called()
    assert kb.graph_search_for(None, None) is None
    assert kb.graph_search_for(object(), None) is None


def test_query_engine_per_domain(monkeypatch):
    policy = Mock(return_value="policy-engine")
    oncall_builder = Mock(return_value="oncall-engine")
    monkeypatch.setattr(kb.services, "build_query_engine", policy)
    monkeypatch.setattr(kb.graph_retrieval, "build_oncall_query_engine", oncall_builder)
    monkeypatch.setattr(kb.storage, "KB_DOMAIN", "policy")
    assert kb.build_query_engine("index", "llm", "reranker", "graph") == "policy-engine"
    assert policy.call_args.kwargs == {"advanced": True, "reranker": "reranker"}
    monkeypatch.setattr(kb.storage, "KB_DOMAIN", "oncall")
    assert kb.build_query_engine("index", "llm", "reranker", "graph") == "oncall-engine"
    # Without a graph the requested mode falls back to hybrid.
    assert oncall_builder.call_args.args[3] == "hybrid"
    search = object()
    kb.build_query_engine("index", "llm", "reranker", "graph", search)
    assert oncall_builder.call_args.args[3:] == ("graph", search)
