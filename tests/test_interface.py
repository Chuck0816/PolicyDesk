"""Verify the web demo without databases, embedding downloads, or paid inference."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import streamlit as st
from llama_index.core.schema import NodeWithScore, TextNode
from streamlit.testing.v1 import AppTest

import app as web
from ui_text import TEXT, translate

APP = Path(__file__).resolve().parents[1] / "app.py"
ZH, EN = TEXT["zh"], TEXT["en"]
DOCUMENT = {
    "doc_id": "pdf-1",
    "file_name": "leave_policy.pdf",
    "doc_type": "pdf",
    "graph_relations": 0,
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
    monkeypatch.setattr("storage.KB_DOMAIN", "policy")
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
    monkeypatch.setattr("kb.list_documents", Mock(side_effect=lambda _: list(documents)))

    def ingest(pdf, context, collection, *, work_dir, progress=None):
        for step in ("parse", "describe", "index"):
            progress(step)
        return "pdf-3", 3, []

    ingest_mock = Mock(side_effect=ingest)
    monkeypatch.setattr("pdf_ingestion.ingest_pdf", ingest_mock)

    def delete(context, doc_id, graph_store=None):
        documents[:] = [item for item in documents if item["doc_id"] != doc_id]

    delete_mock = Mock(side_effect=delete)
    monkeypatch.setattr("kb.delete_document", delete_mock)
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
    assert result["runs"][0]["answer"] == "L3 员工每晚上限 700 元。"
    # Tables are shown as plain "cell | cell" text, never as raw HTML.
    assert result["runs"][0]["sources"][0]["text"] == "Table: 表 1\n职级 | 上限\nL3 | 700"
    text = markdown_text(app)
    assert 'class="pd-phase done">撰写回答' in text
    assert "差旅制度.pdf · 第 1 页" in text and ZH["kind_table"] in text
    assert "<table>" not in text
    assert any("相关度 0.87" in item.value for item in app.markdown)
    assert backend.builder.call_args.kwargs["advanced"] is True
    app.run()
    assert any(result["runs"][0]["answer"] in item.value for item in app.markdown)
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
    app.session_state["result"] = {"question": "Old", "runs": []}
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
    assert app.session_state["result"]["runs"][0]["answer"] == "OK"


def test_missing_index_is_reported_before_models_load(backend):
    backend.context.index_store.get_index_struct.return_value = None
    app = ask(run_app())
    assert not app.exception
    assert "无法完成提问" in app.error[0].value
    backend.models.assert_not_called()


def test_database_outage_is_reported(backend, monkeypatch):
    monkeypatch.setattr("kb.list_documents", Mock(side_effect=RuntimeError("down")))
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
    monkeypatch.setattr("storage.KB_DOMAIN", "policy")
    monkeypatch.setattr(web, "load_models", Mock(return_value=(object(), object())))
    monkeypatch.setattr(web, "graph_resources", Mock(return_value=(None, None, None)))
    monkeypatch.setattr(web.storage, "connect_storage", Mock(return_value=("ctx", "col")))
    seen = {}

    def ingest(pdf, context, collection, *, work_dir, progress=None):
        seen.update(name=pdf.name, data=pdf.read_bytes(), progress=progress)
        return "pdf-9", 1, []

    monkeypatch.setattr("pdf_ingestion.ingest_pdf", ingest)
    marker = object()
    assert web.ingest_upload("../报销 制度.pdf", b"%PDF", progress=marker) == (
        "pdf-9",
        1,
        {"skipped": []},
    )
    assert seen == {"name": "报销 制度.pdf", "data": b"%PDF", "progress": marker}


def test_plain_text_flattens_tables_and_drops_markup():
    assert web.plain_text("Note <b>bold</b>") == "Note\nbold"
    assert web.plain_text("no markup") == "no markup"
    assert web.plain_text(TABLE) == "Table: 表 1\n职级 | 上限\nL3 | 700"


# --- On-call domain ---------------------------------------------------------------------

STRUCTURED = '```json\n{"root_cause_service": "redis-cart", "owner_team": "team-payments"}\n```'
GRAPH_PATH = "checkout -DEPENDS_ON-> cart -DEPENDS_ON-> redis-cart"


def oncall_text(key, **values):
    return translate("zh", key, domain="oncall", **values)


@pytest.fixture
def oncall(backend, monkeypatch):
    monkeypatch.setattr("storage.KB_DOMAIN", "oncall")
    graph_store = object()
    catalog = SimpleNamespace(
        components={"checkout": 1, "cart": 2}, teams={"team-payments": 1}, alerts={"a": 1}
    )
    connect_graph = Mock(return_value=(graph_store, catalog))
    monkeypatch.setattr("kb.connect_graph", connect_graph)
    monkeypatch.setattr("kb.build_extraction", Mock(return_value=("extractor", "cache")))

    def engine_for(index, llm, reranker, mode, graph_search):
        engine = Mock()
        sources = [
            NodeWithScore(
                node=TextNode(
                    text="Restart the cart pods.",
                    metadata={
                        "file_name": "cart.md",
                        "section": "Mitigation",
                        "content_type": "text",
                    },
                ),
                score=0.9,
            )
        ]
        if mode != "hybrid":
            sources.append(
                NodeWithScore(
                    node=TextNode(text=GRAPH_PATH, metadata={"content_type": "graph_path"}),
                    score=1.0,
                )
            )
        engine.query.side_effect = lambda question: SimpleNamespace(
            response_gen=iter([f"Answer from {mode}.\n\n", STRUCTURED]), source_nodes=sources
        )
        return engine

    builder = Mock(side_effect=engine_for)
    monkeypatch.setattr("graph_retrieval.build_oncall_query_engine", builder)
    imported = SimpleNamespace(components={"a": 1, "b": 2, "c": 3}, teams={"t": 1}, alerts={})
    import_catalog = Mock(return_value=(imported, 3))
    monkeypatch.setattr("kb.import_catalog", import_catalog)
    backend.documents[:] = [
        dict(
            DOCUMENT,
            doc_id="md-1",
            file_name="INC-001.md",
            doc_type="postmortem",
            graph_relations=6,
        )
    ]
    return SimpleNamespace(
        builder=builder,
        graph_store=graph_store,
        catalog=catalog,
        connect_graph=connect_graph,
        import_catalog=import_catalog,
    )


def test_oncall_answer_splits_evidence_graph_paths_and_hidden_json(backend, oncall):
    app = ask(run_app(), "checkout 延迟升高，根因可能是什么？")
    assert not app.exception
    run = app.session_state["result"]["runs"][0]
    assert run["mode"] == "hybrid+graph"
    assert run["answer"] == "Answer from hybrid+graph."
    assert run["structured"] == {"root_cause_service": "redis-cart", "owner_team": "team-payments"}
    assert run["timing"]["total_s"] is not None and "llm_calls" in run["tokens"]
    text = markdown_text(app)
    assert oncall_text("sources_label", count=1) in text
    assert oncall_text("graph_paths_label", count=1) in text
    assert GRAPH_PATH.replace(">", "&gt;") in text
    # The JSON block is only inside the evaluation expander, not in the visible answer.
    assert "root_cause_service" not in text
    assert any(item.label == oncall_text("structured_label") for item in app.expander)
    assert any("检索" in item.value and "LLM" in item.value for item in app.markdown)
    assert oncall.builder.call_args.args[3] == "hybrid+graph"
    assert oncall.builder.call_args.args[4].graph_store is oncall.graph_store


def test_oncall_mode_switch_uses_the_selected_mode(backend, oncall):
    app = run_app()
    app = app.segmented_control(key="mode").set_value("graph").run()
    app = ask(app, "谁负责 checkout？")
    assert not app.exception
    assert oncall.builder.call_args.args[3] == "graph"
    assert app.session_state["result"]["runs"][0]["answer"] == "Answer from graph."


def test_oncall_compare_runs_hybrid_and_hybrid_graph_side_by_side(backend, oncall):
    app = run_app().toggle(key="compare").set_value(True).run()
    assert app.segmented_control(key="mode").disabled
    app = ask(app, "checkout 故障影响哪些服务？")
    assert not app.exception
    runs = app.session_state["result"]["runs"]
    assert [run["mode"] for run in runs] == ["hybrid", "hybrid+graph"]
    assert [run["answer"] for run in runs] == ["Answer from hybrid.", "Answer from hybrid+graph."]
    assert not runs[0]["sources"][-1]["kind"] == "graph_path"
    assert runs[1]["sources"][-1]["kind"] == "graph_path"
    # The saved result keeps both columns after a rerun.
    app.run()
    text = markdown_text(app)
    assert oncall_text("mode_hybrid") in text and oncall_text("mode_hybrid+graph") in text


def test_oncall_without_neo4j_falls_back_to_hybrid(backend, oncall, monkeypatch):
    monkeypatch.setattr("kb.connect_graph", Mock(side_effect=RuntimeError("refused")))
    app = ask(run_app(), "谁负责 checkout？")
    assert not app.exception
    assert any("refused" in item.value for item in app.caption)
    assert app.session_state["result"]["runs"][0]["mode"] == "hybrid"


def test_oncall_documents_show_types_and_relations_and_import_the_catalog(backend, oncall):
    app = open_documents(run_app())
    assert not app.exception
    text = markdown_text(app)
    assert "INC-001.md" in text and oncall_text("type_postmortem") in text
    assert oncall_text("status_relations", count=6) in text
    assert any(
        oncall_text("catalog_status", services=2, alerts=1) in item.value for item in app.caption
    )
    catalog_uploader = next(item for item in app.file_uploader if item.key == "catalog-0")
    catalog_uploader.upload("service_catalog.yaml", b"services: []", "application/x-yaml").run()
    app = app.button(key="import").click().run()
    assert not app.exception
    args = oncall.import_catalog.call_args.args
    assert args[:2] == ("services: []", "") and args[4] is oncall.graph_store
    text = markdown_text(app)
    assert oncall_text("catalog_imported", services=3, teams=1, alerts=0) in text


def test_oncall_markdown_upload_passes_the_type_and_reports_extraction(
    backend, oncall, monkeypatch
):
    def ingest(path, context, collection, **kwargs):
        kwargs["progress"]("extract")
        return "md-2", 2, {"chunks": 2, "empty_chunks": 0, "dropped": ["billing"], "relations": 5}

    add = Mock(side_effect=ingest)
    monkeypatch.setattr("kb.add_document", add)
    app = open_documents(run_app())
    app = app.segmented_control(key="md_type").set_value("postmortem").run()
    upload = next(item for item in app.file_uploader if item.key == "upload-0")
    upload.upload("INC-002.md", b"---\nincident_id: INC-002\n---\n", "text/markdown").run()
    app = button(app, oncall_text("add_button")).click().run()
    assert not app.exception
    kwargs = add.call_args.kwargs
    assert add.call_args.args[0].name == "INC-002.md"
    assert kwargs["doc_type"] == "postmortem"
    assert kwargs["graph_store"] is oncall.graph_store and kwargs["catalog"] is oncall.catalog
    assert kwargs["extraction"] == ("extractor", "cache")
    text = markdown_text(app)
    assert oncall_text("added", name="INC-002.md", count=2) in text
    assert "billing" in text


def test_oncall_catalog_row_cannot_be_deleted(backend, oncall):
    backend.documents.insert(
        0,
        dict(
            DOCUMENT,
            doc_id="catalog",
            file_name="service_catalog.yaml",
            doc_type="service_catalog",
            services=12,
        ),
    )
    app = open_documents(run_app())
    assert not app.exception
    keys = {item.key for item in app.button}
    assert "delete-catalog" not in keys and "delete-md-1" in keys


def test_oncall_alerts_alone_update_the_imported_catalog(backend, oncall, monkeypatch):
    source = Mock(return_value={"catalog": "services: [stored]", "alerts": "alerts: []"})
    monkeypatch.setattr("kb.service_catalog.catalog_source", source)
    app = open_documents(run_app())
    assert app.button(key="import").disabled
    alerts = next(item for item in app.file_uploader if item.key == "alerts-0")
    alerts.upload("alerts.yaml", b"alerts: [new]", "application/x-yaml").run()
    assert not app.button(key="import").disabled
    assert any(oncall_text("alerts_only") in item.value for item in app.caption)
    app = app.button(key="import").click().run()
    assert not app.exception
    args = oncall.import_catalog.call_args.args
    assert args[:2] == ("services: [stored]", "alerts: [new]")
    assert args[4] is oncall.graph_store


def test_oncall_alerts_alone_need_a_catalog_first(backend, oncall, monkeypatch):
    monkeypatch.setattr("kb.connect_graph", Mock(return_value=(object(), None)))
    app = open_documents(run_app())
    alerts = next(item for item in app.file_uploader if item.key == "alerts-0")
    alerts.upload("alerts.yaml", b"alerts: []", "application/x-yaml").run()
    assert app.button(key="import").disabled
    assert any(oncall_text("catalog_needs_file") in item.value for item in app.caption)
