"""Verify the web demo without databases, embedding downloads, or paid inference."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import streamlit as st
from llama_index.core.schema import NodeWithScore, TextNode
from streamlit.testing.v1 import AppTest

import app as web
from ui_text import TEXT

APP = Path(__file__).resolve().parents[1] / "app.py"
ZH, EN = TEXT["zh"], TEXT["en"]
DOCUMENT = {
    "doc_id": "pdf-1",
    "file_name": "leave_policy.pdf",
    "uploaded_at": "2026-11-12T08:30:00+00:00",
    "skipped_figures": 0,
    "chunks": 4,
}
TABLE = (
    "Table: 表 1\n"
    "<table><tr><td>职级</td><td>上限</td></tr><tr><td>L3</td><td>700</td></tr></table>"
)


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
    documents = [
        dict(DOCUMENT),
        dict(DOCUMENT, doc_id="pdf-2", file_name="差旅制度.pdf", skipped_figures=1, chunks=9),
    ]
    monkeypatch.setattr("pdf_ingestion.list_pdfs", Mock(side_effect=lambda _: list(documents)))

    def ingest(pdf, context, collection, *, work_dir, progress=None):
        for step in ("parse", "describe", "index"):
            progress(step)
        return "pdf-3", 3, []

    ingest_mock = Mock(side_effect=ingest)
    monkeypatch.setattr("pdf_ingestion.ingest_pdf", ingest_mock)

    def delete(context, doc_id):
        documents[:] = [item for item in documents if item["doc_id"] != doc_id]

    delete_mock = Mock(side_effect=delete)
    monkeypatch.setattr("pdf_ingestion.delete_pdf", delete_mock)
    engine = Mock()
    engine.query.side_effect = lambda question: SimpleNamespace(
        response_gen=iter(["L3 员工每晚上限 ", "700 元。"]),
        source_nodes=[
            NodeWithScore(
                node=TextNode(
                    text=TABLE,
                    metadata={"file_name": "差旅制度.pdf", "page": 1, "content_type": "table"},
                ),
                score=0.87,
            )
        ],
    )
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
        ingest=ingest_mock,
        delete=delete_mock,
    )
    st.cache_resource.clear()


def run_app():
    return AppTest.from_file(str(APP), default_timeout=15).run()


def button(app, label):
    return next(item for item in app.button if item.label == label)


def markdown_text(app):
    return "\n".join(item.value for item in app.markdown)


def ask(app, question="L3 住宿上限是多少？"):
    app.text_area(key="question").set_value(question)
    return button(app, ZH["ask_button"]).click().run()


def open_documents(app):
    return app.segmented_control(key="view").set_value("docs").run()


def test_empty_knowledge_base_points_to_documents_without_loading_models(backend):
    backend.documents.clear()
    app = run_app()
    assert not app.exception
    assert any(ZH["empty_ask"] in item.value for item in app.markdown)
    assert button(app, ZH["ask_button"]).disabled
    app = button(app, ZH["go_docs"]).click().run()
    assert app.segmented_control(key="view").value == "docs"
    assert any(ZH["no_docs"] in item.value for item in app.info)
    backend.models.assert_not_called()


def test_empty_question_does_not_load_models(backend):
    app = button(run_app(), ZH["ask_button"]).click().run()
    assert ZH["enter_question"] in app.warning[0].value
    backend.models.assert_not_called()


def test_question_shows_phases_source_cards_answer_and_excerpts(backend):
    app = ask(run_app())
    assert not app.exception
    backend.engine.query.assert_called_once_with("L3 住宿上限是多少？")
    result = app.session_state["result"]
    assert result["answer"] == "L3 员工每晚上限 700 元。"
    # Tables are shown as plain "cell | cell" text, never as raw HTML.
    assert result["sources"][0]["text"] == "Table: 表 1\n职级 | 上限\nL3 | 700"
    text = markdown_text(app)
    assert 'class="pd-phase done">撰写回答' in text
    assert "差旅制度.pdf · 第 1 页" in text and ZH["kind_table"] in text
    assert "<table>" not in text
    assert any("相关度 0.87" in item.value for item in app.markdown)
    assert backend.builder.call_args.kwargs["advanced"] is True
    app.run()
    assert any(result["answer"] in item.value for item in app.markdown)
    assert backend.engine.query.call_count == 1


def test_example_question_is_asked_immediately(backend):
    app = run_app()
    app = app.button(key="example-0").click().run()
    assert not app.exception
    backend.engine.query.assert_called_once_with(ZH["examples"][0])


def test_documents_table_filters_and_shows_status(backend):
    app = open_documents(run_app())
    assert not app.exception
    text = markdown_text(app)
    assert "leave_policy.pdf" in text and "差旅制度.pdf" in text
    assert ZH["status_ok"] in text and ZH["status_skipped"].format(count=1) in text
    assert "2026-11-12 08:30 UTC" in text
    app = app.segmented_control(key="doc_filter").set_value("skipped").run()
    text = markdown_text(app)
    assert "差旅制度.pdf" in text and "leave_policy.pdf" not in text


def test_upload_reports_steps_and_skipped_figures_then_rebuilds_engine(backend):
    app = ask(run_app())
    assert backend.builder.call_count == 1

    def ingest(pdf, context, collection, *, work_dir, progress=None):
        progress("parse")
        return "pdf-3", 3, [{"page": 5, "caption": "图 2", "reason": "RuntimeError: quota"}]

    backend.ingest.side_effect = ingest
    app = open_documents(app)
    app.file_uploader[0].upload("信息安全.pdf", b"%PDF-1.7", "application/pdf").run()
    app = button(app, ZH["add_button"]).click().run()
    assert not app.exception
    pdf = backend.ingest.call_args.args[0]
    assert pdf.name == "信息安全.pdf"
    assert callable(backend.ingest.call_args.kwargs["progress"])
    assert [call.kwargs.get("create", False) for call in backend.connect.call_args_list] == [
        False,
        True,
    ]
    text = markdown_text(app)
    assert "已添加 信息安全.pdf（3 段）" in text
    assert "第 5 页 · 图 2 — RuntimeError: quota" in text
    assert "result" not in app.session_state
    app = app.segmented_control(key="view").set_value("ask").run()
    ask(app)
    assert backend.builder.call_count == 2
    backend.models.assert_called_once()


def test_failed_upload_shows_the_reason(backend):
    backend.ingest.side_effect = ValueError("信息安全.pdf is already in the knowledge base.")
    app = open_documents(run_app())
    app.file_uploader[0].upload("信息安全.pdf", b"%PDF-1.7", "application/pdf").run()
    app = button(app, ZH["add_button"]).click().run()
    assert not app.exception
    assert "无法添加 信息安全.pdf。" in markdown_text(app)
    assert "already in the knowledge base" in markdown_text(app)


def test_delete_needs_confirmation_and_can_be_cancelled(backend):
    app = open_documents(run_app())
    app = app.button(key="delete-pdf-1").click().run()
    assert any("确定从知识库删除 leave_policy.pdf" in item.value for item in app.markdown)
    app = app.button(key="cancel-pdf-1").click().run()
    backend.delete.assert_not_called()
    app = app.button(key="delete-pdf-1").click().run()
    app = app.button(key="really-delete-pdf-1").click().run()
    assert not app.exception
    assert backend.delete.call_args.args[1] == "pdf-1"
    text = markdown_text(app)
    assert "已删除 leave_policy.pdf。" in text
    assert "leave_policy.pdf</div>" not in text


def test_language_switch_changes_interface_text(backend):
    app = run_app().segmented_control(key="language").set_value("en").run()
    assert not app.exception
    assert any(EN["ask_title"] in item.value for item in app.markdown)
    assert button(app, EN["ask_button"])
    app = open_documents(app)
    assert any(EN["docs_title"] in item.value for item in app.markdown)


def test_backend_failure_removes_stale_answer_and_allows_retry(backend):
    app = run_app()
    app.session_state["result"] = {"question": "Old", "answer": "Stale", "sources": []}
    backend.engine.query.side_effect = RuntimeError("Backend unavailable")
    app = ask(app)
    assert not app.exception
    assert ZH["query_failed"] in app.error[0].value
    assert "result" not in app.session_state
    backend.engine.query.side_effect = lambda question: SimpleNamespace(
        response_gen=iter(["OK"]), source_nodes=[]
    )
    app = button(app, ZH["ask_button"]).click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["result"]["answer"] == "OK"


def test_missing_index_is_reported_before_models_load(backend):
    backend.context.index_store.get_index_struct.return_value = None
    app = ask(run_app())
    assert not app.exception
    assert "无法完成提问" in app.error[0].value
    backend.models.assert_not_called()


def test_database_outage_is_reported(backend, monkeypatch):
    monkeypatch.setattr("pdf_ingestion.list_pdfs", Mock(side_effect=RuntimeError("down")))
    app = run_app()
    assert not app.exception
    assert ZH["db_error"] in app.error[0].value
    assert button(app, ZH["ask_button"]).disabled


def test_empty_provider_stream_shows_actionable_error_and_no_result(backend):
    import services

    backend.engine.query.side_effect = lambda question: SimpleNamespace(
        response_gen=services.validated_answer_stream(iter([""])), source_nodes=[]
    )
    app = ask(run_app())
    assert not app.exception
    assert "account quota" in app.error[0].value
    assert "result" not in app.session_state


def test_uploads_keep_their_file_name(monkeypatch):
    monkeypatch.setattr(web, "load_models", Mock())
    monkeypatch.setattr(web.storage, "connect_storage", Mock(return_value=("ctx", "col")))
    seen = {}

    def ingest(pdf, context, collection, *, work_dir, progress=None):
        seen.update(name=pdf.name, data=pdf.read_bytes(), progress=progress)
        return "pdf-9", 1, []

    monkeypatch.setattr(web.pdf_ingestion, "ingest_pdf", ingest)
    marker = object()
    assert web.ingest_upload("../报销 制度.pdf", b"%PDF", progress=marker) == ("pdf-9", 1, [])
    assert seen == {"name": "报销 制度.pdf", "data": b"%PDF", "progress": marker}


def test_plain_text_flattens_tables_and_drops_markup():
    assert web.plain_text("Note <b>bold</b>") == "Note\nbold"
    assert web.plain_text("no markup") == "no markup"
    assert web.plain_text(TABLE) == "Table: 表 1\n职级 | 上限\nL3 | 700"
