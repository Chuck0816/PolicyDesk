"""Verify the web demo without databases, embedding downloads, or paid inference."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import streamlit as st
from llama_index.core.schema import NodeWithScore, TextNode
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
DOCUMENT = {
    "doc_id": "pdf-1",
    "file_name": "leave_policy.pdf",
    "uploaded_at": "2026-11-12T08:30:00+00:00",
    "skipped_figures": 1,
    "chunks": 4,
}


@pytest.fixture
def backend(monkeypatch):
    st.cache_resource.clear()
    models = Mock(return_value=(object(), object()))
    monkeypatch.setattr("base_llm.create_models", models)
    monkeypatch.setattr("services.configure_settings", Mock())
    context = SimpleNamespace(index_store=Mock())
    connect = Mock(return_value=(context, Mock()))
    monkeypatch.setattr("storage.connect_storage", connect)
    monkeypatch.setattr("storage.connect_docstore", Mock())
    monkeypatch.setattr("llama_index.core.load_index_from_storage", Mock())
    reranker = Mock(return_value=object())
    monkeypatch.setattr("base_llm.create_reranker", reranker)
    documents = [dict(DOCUMENT)]
    monkeypatch.setattr("pdf_ingestion.list_pdfs", Mock(side_effect=lambda _: list(documents)))
    ingest = Mock(return_value=("pdf-2", 3, []))
    monkeypatch.setattr("pdf_ingestion.ingest_pdf", ingest)
    delete = Mock(side_effect=lambda *_: documents.clear())
    monkeypatch.setattr("pdf_ingestion.delete_pdf", delete)
    response = SimpleNamespace(
        response_gen=iter(["The department head ", "approves four working days of leave."]),
        source_nodes=[
            NodeWithScore(
                node=TextNode(
                    text="Leave exceeding three working days requires department head approval.",
                    metadata={"file_name": "leave_policy.pdf", "page": 2, "content_type": "table"},
                ),
                score=0.87,
            )
        ],
    )
    engine = Mock()
    engine.query.return_value = response
    builder = Mock(return_value=engine)
    monkeypatch.setattr("services.build_query_engine", builder)
    yield SimpleNamespace(
        models=models,
        engine=engine,
        context=context,
        connect=connect,
        builder=builder,
        reranker=reranker,
        documents=documents,
        ingest=ingest,
        delete=delete,
    )
    st.cache_resource.clear()


def run_app():
    return AppTest.from_file(str(APP), default_timeout=15).run()


def button(app, label):
    return next(item for item in app.button if item.label == label)


def ask(app, question="Who approves leave?"):
    app.text_area[0].set_value(question)
    return button(app, "Ask PolicyDesk →").click().run()


def test_empty_knowledge_base_asks_for_a_pdf_and_loads_no_models(backend):
    backend.documents.clear()
    app = run_app()
    assert not app.exception
    assert any("Add a policy PDF" in item.value for item in app.info)
    assert button(app, "Ask PolicyDesk →").disabled
    assert button(app, "Add to knowledge base").disabled
    backend.models.assert_not_called()


def test_empty_question_does_not_load_models(backend):
    app = button(run_app(), "Ask PolicyDesk →").click().run()
    assert "Enter a question" in app.warning[0].value
    backend.models.assert_not_called()


def test_document_list_shows_details_and_skipped_figures(backend):
    app = run_app()
    assert any("leave_policy.pdf" in item.value for item in app.sidebar.markdown)
    assert any(
        "4 chunks · added 2026-11-12 08:30 UTC · 1 figure(s) skipped" in item.value
        for item in app.sidebar.caption
    )


def test_question_streams_answer_with_page_and_type(backend):
    app = ask(run_app())
    assert not app.exception
    backend.engine.query.assert_called_once_with("Who approves leave?")
    result = app.session_state["result"]
    assert result["answer"] == "The department head approves four working days of leave."
    assert result["sources"][0]["name"] == "leave_policy.pdf"
    assert any(item.value == "Page 2 · Table" for item in app.caption)
    assert any("Reranker score: 0.8700" in item.value for item in app.caption)
    assert backend.builder.call_args.kwargs["advanced"] is True
    app.run()
    assert any(result["answer"] in item.value for item in app.markdown)
    assert backend.engine.query.call_count == 1


def test_upload_adds_pdf_reports_skipped_figures_and_rebuilds_engine(backend):
    app = ask(run_app())
    assert backend.builder.call_count == 1
    backend.ingest.return_value = (
        "pdf-2",
        3,
        [{"page": 5, "caption": "图 2 审批流程", "reason": "RuntimeError: quota"}],
    )
    app.file_uploader[0].upload("差旅制度.pdf", b"%PDF-1.7", "application/pdf").run()
    app = button(app, "Add to knowledge base").click().run()
    assert not app.exception
    pdf, _, _ = backend.ingest.call_args.args
    assert pdf.name == "差旅制度.pdf"
    # Only adding may create the Chroma collection of a new knowledge base.
    assert [call.kwargs.get("create", False) for call in backend.connect.call_args_list] == [
        False,
        True,
    ]
    assert any("Added 差旅制度.pdf (3 chunks)" in item.value for item in app.success)
    assert any("Page 5 (图 2 审批流程): RuntimeError: quota" in item.value for item in app.warning)
    assert "result" not in app.session_state
    ask(app)
    assert backend.builder.call_count == 2
    backend.models.assert_called_once()


def test_failed_upload_shows_the_reason(backend):
    backend.ingest.side_effect = ValueError(
        "差旅制度.pdf is already in the knowledge base (pdf-2)."
    )
    app = run_app()
    app.file_uploader[0].upload("差旅制度.pdf", b"%PDF-1.7", "application/pdf").run()
    app = button(app, "Add to knowledge base").click().run()
    assert not app.exception
    assert any("already in the knowledge base" in item.value for item in app.error)


def test_delete_needs_confirmation_and_can_be_cancelled(backend):
    app = button(run_app(), "Delete...").click().run()
    assert "Delete this PDF from the knowledge base?" in [item.value for item in app.caption]
    app = button(app, "Cancel").click().run()
    backend.delete.assert_not_called()
    app = button(app, "Delete...").click().run()
    app = button(app, "Delete").click().run()
    assert not app.exception
    assert backend.delete.call_args.args[1] == "pdf-1"
    assert any("Deleted leave_policy.pdf" in item.value for item in app.success)
    assert button(app, "Ask PolicyDesk →").disabled


def test_backend_failure_removes_stale_answer_and_allows_retry(backend):
    app = run_app()
    app.session_state["result"] = {"question": "Old", "answer": "Stale", "sources": []}
    backend.engine.query.side_effect = RuntimeError("Backend unavailable")
    app = ask(app)
    assert not app.exception
    assert "Could not complete" in app.error[0].value
    assert "result" not in app.session_state
    backend.engine.query.side_effect = None
    app = button(app, "Ask PolicyDesk →").click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["result"]["answer"]


def test_missing_index_is_reported_before_models_load(backend):
    backend.context.index_store.get_index_struct.return_value = None
    app = ask(run_app())
    assert not app.exception
    assert "Could not complete" in app.error[0].value
    backend.models.assert_not_called()


def test_database_outage_is_reported(backend, monkeypatch):
    monkeypatch.setattr("pdf_ingestion.list_pdfs", Mock(side_effect=RuntimeError("down")))
    app = run_app()
    assert not app.exception
    assert "Could not reach the database services" in app.error[0].value
    assert button(app, "Ask PolicyDesk →").disabled


def test_empty_provider_stream_shows_actionable_error_and_no_result(backend):
    import services

    backend.engine.query.return_value.response_gen = services.validated_answer_stream(iter([""]))
    app = ask(run_app())
    assert not app.exception
    assert "account quota" in app.error[0].value
    assert "result" not in app.session_state
