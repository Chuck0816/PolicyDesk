"""Verify the web demo without databases, embedding downloads, or paid inference."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import streamlit as st
from llama_index.core.schema import NodeWithScore, TextNode
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def backend(monkeypatch):
    st.cache_resource.clear()
    models = Mock(return_value=(object(), object()))
    monkeypatch.setattr("base_llm.create_models", models)
    monkeypatch.setattr("services.configure_settings", Mock())
    context = SimpleNamespace(index_store=Mock())
    monkeypatch.setattr("storage.connect_storage", Mock(return_value=(context, object())))
    monkeypatch.setattr("llama_index.core.load_index_from_storage", Mock())
    reranker = Mock(return_value=object())
    monkeypatch.setattr("base_llm.create_reranker", reranker)
    response = SimpleNamespace(
        response_gen=iter(["The department head ", "approves four working days of leave."]),
        source_nodes=[
            NodeWithScore(
                node=TextNode(
                    text="Leave exceeding three working days requires department head approval.",
                    metadata={
                        "file_name": "leave_policy.txt",
                        "policy_id": "HR-LEAVE-001",
                        "version": "1.0",
                        "category": "leave",
                    },
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
        models=models, engine=engine, context=context, builder=builder, reranker=reranker
    )
    st.cache_resource.clear()


def test_opening_page_and_empty_submit_do_not_load_models(backend):
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not app.exception
    backend.models.assert_not_called()
    app.button[-1].click().run()
    assert "Enter a question" in app.warning[0].value
    backend.models.assert_not_called()


def test_example_streams_answer_and_shows_sources(backend):
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    app.button[0].click().run()
    assert app.text_area[0].value == "Who approves four working days of leave?"
    app.button[-1].click().run()
    assert not app.exception
    backend.engine.query.assert_called_once_with("Who approves four working days of leave?")
    result = app.session_state["result"]
    assert result["answer"] == "The department head approves four working days of leave."
    assert result["sources"][0]["name"] == "leave_policy.txt"
    assert any("Reranker score: 0.8700" in item.value for item in app.caption)
    assert any("HR-LEAVE-001 · Version 1.0 · Leave" in item.value for item in app.caption)
    assert backend.builder.call_args.kwargs["advanced"] is True
    assert any("Leave exceeding three" in item.value for item in app.text)
    app.run()
    assert not app.exception
    assert any(result["answer"] in item.value for item in app.markdown)
    assert backend.engine.query.call_count == 1


def test_backend_failure_removes_stale_answer_and_allows_retry(backend):
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    app.session_state["result"] = {"question": "Old", "answer": "Stale", "sources": []}
    backend.engine.query.side_effect = RuntimeError("Backend unavailable")
    app.text_area[0].set_value("Who approves leave?")
    app.button[-1].click().run()
    assert not app.exception
    assert "Could not complete" in app.error[0].value
    assert "result" not in app.session_state
    backend.engine.query.side_effect = None
    app.button[-1].click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["result"]["answer"]


def test_missing_index_is_reported_before_models_load(backend):
    backend.context.index_store.get_index_struct.return_value = None
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    app.text_area[0].set_value("Who approves leave?")
    app.button[-1].click().run()
    assert not app.exception
    assert "Could not complete" in app.error[0].value
    backend.models.assert_not_called()


def test_scope_change_clears_answer_and_reuses_model_resources(backend):
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    app.text_area[0].set_value("Who approves leave?")
    app.button[-1].click().run()
    assert "result" in app.session_state
    assert backend.builder.call_args.kwargs["category"] is None
    app.selectbox[0].set_value("salary").run()
    assert not app.exception
    assert "result" not in app.session_state
    assert backend.engine.query.call_count == 1
    app.button[-1].click().run()
    assert backend.builder.call_args.kwargs["category"] == "salary"
    assert backend.builder.call_count == 2
    backend.models.assert_called_once()
    backend.reranker.assert_called_once()
    app.selectbox[0].set_value(None).run()
    app.button[-1].click().run()
    assert backend.builder.call_count == 2
    assert not app.exception


def test_empty_provider_stream_shows_actionable_error_and_no_result(backend):
    import services

    backend.engine.query.return_value.response_gen = services.validated_answer_stream(iter([""]))
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    app.text_area[0].set_value("Who approves leave?")
    app.button[-1].click().run()
    assert not app.exception
    assert "account quota" in app.error[0].value
    assert "result" not in app.session_state
