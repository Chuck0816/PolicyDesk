"""CLI documents, catalog import, modes, timing, and conversation keys without services."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main

DOCUMENT = {
    "doc_id": "pdf-1",
    "file_name": "leave.pdf",
    "doc_type": "pdf",
    "uploaded_at": "2026-11-12T08:30:00+00:00",
    "skipped_figures": 0,
    "graph_relations": 0,
    "chunks": 4,
}


def args(command, target=None, **options):
    defaults = {
        "content_list": None,
        "doc_type": None,
        "alerts": None,
        "mode": "hybrid+graph",
        "session": "demo",
    }
    return Namespace(command=command, target=target, **{**defaults, **options})


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(main.storage, "KB_DOMAIN", "policy")
    context = SimpleNamespace(index_store=Mock())
    monkeypatch.setattr(main.storage, "connect_storage", Mock(return_value=(context, Mock())))
    monkeypatch.setattr(main.storage, "connect_docstore", Mock())
    monkeypatch.setattr(main.kb, "list_documents", Mock(return_value=[DOCUMENT]))
    monkeypatch.setattr(main.kb, "add_document", Mock(return_value=("pdf-2", 3, {"skipped": []})))
    monkeypatch.setattr(main.kb, "delete_document", Mock())
    monkeypatch.setattr(main.kb, "import_catalog", Mock())
    monkeypatch.setattr(main, "create_models", Mock(return_value=(object(), object())))
    monkeypatch.setattr(main, "create_reranker", Mock(return_value=object()))
    monkeypatch.setattr(main, "load_index_from_storage", Mock())
    monkeypatch.setattr(main.functions, "configure_settings", Mock())
    engine = Mock()
    engine.query.return_value = SimpleNamespace(response_gen=iter(["Pay", "roll"]), source_nodes=[])
    builder = Mock(return_value=engine)
    monkeypatch.setattr(main.functions, "build_query_engine", builder)
    memory = Mock()
    monkeypatch.setattr(main.functions, "build_chat_memory", memory)
    monkeypatch.setattr(main.storage, "connect_chat_store", Mock())
    chat = SimpleNamespace(chat_history=[])
    monkeypatch.setattr(main.CondenseQuestionChatEngine, "from_defaults", Mock(return_value=chat))
    monkeypatch.setattr("builtins.input", lambda _: "/exit")
    return SimpleNamespace(builder=builder, memory=memory, engine=engine, context=context)


@pytest.fixture
def oncall(monkeypatch, backend):
    """The on-call domain with a graph store and catalog stand-ins."""
    monkeypatch.setattr(main.storage, "KB_DOMAIN", "oncall")
    graph_store, catalog = object(), SimpleNamespace(components={}, teams={}, alerts={})
    monkeypatch.setattr(main.kb, "connect_graph", Mock(return_value=(graph_store, catalog)))
    monkeypatch.setattr(main.kb, "build_extraction", Mock(return_value=("extractor", "cache")))
    graph_engine = Mock()
    graph_engine.query.return_value = SimpleNamespace(response_gen=iter(["ok"]), source_nodes=[])
    monkeypatch.setattr(
        main.graph_retrieval, "build_oncall_query_engine", Mock(return_value=graph_engine)
    )
    return SimpleNamespace(graph_store=graph_store, catalog=catalog, engine=graph_engine)


def test_chat_keeps_one_history_per_session(backend):
    main.run(args("chat", session="employee"))
    assert backend.memory.call_args.args[1] == f"{main.storage.KB_NAME}:chat:employee"
    assert backend.builder.call_args.kwargs["advanced"] is True


def test_ask_streams_answer_and_reports_timing(backend, capsys):
    main.run(args("ask", "When is payroll paid?"))
    backend.engine.query.assert_called_once_with("When is payroll paid?")
    main.storage.connect_storage.assert_called_once_with(main.functions, create=False)
    output = capsys.readouterr().out
    assert "Payroll" in output
    assert "Timing (hybrid): retrieval" in output and "LLM calls" in output


@pytest.mark.parametrize("command", ["ask", "chat"])
def test_questions_need_a_document_before_loading_models(backend, command):
    main.kb.list_documents.return_value = []
    with pytest.raises(ValueError, match="has no documents"):
        main.run(args(command, "Who approves leave?"))
    main.create_models.assert_not_called()
    main.storage.connect_storage.assert_not_called()


@pytest.mark.parametrize("command", ["add", "delete", "ask", "import-catalog"])
def test_missing_target_is_rejected_before_connecting(backend, command):
    with pytest.raises(ValueError, match="Provide"):
        main.run(args(command, "  "))
    main.storage.connect_storage.assert_not_called()


def test_add_creates_storage_and_reports_skipped_figures(backend, capsys):
    main.kb.add_document.return_value = (
        "pdf-2",
        3,
        {"skipped": [{"page": 4, "caption": "图 1", "reason": "RuntimeError: quota"}]},
    )
    main.run(args("add", "travel.pdf"))
    main.storage.connect_storage.assert_called_once_with(main.functions, create=True)
    assert main.kb.add_document.call_args.args[0] == "travel.pdf"
    output = capsys.readouterr().out
    assert "Added pdf-2: 3 chunks" in output
    assert "Skipped figure on page 4 (图 1): RuntimeError: quota" in output


def test_add_can_reuse_a_mineru_content_list(backend, tmp_path):
    content_list = tmp_path / "travel_content_list.json"
    main.run(args("add", "travel.pdf", content_list=str(content_list)))
    parse = main.kb.add_document.call_args.kwargs["parse"]
    assert parse("travel.pdf", tmp_path) == content_list


@pytest.mark.parametrize(
    "command, option, message",
    [
        ("ask", {"content_list": "x.json"}, "only to add"),
        ("ask", {"doc_type": "runbook"}, "only to add"),
        ("add", {"alerts": "alerts.yaml"}, "only to import-catalog"),
    ],
)
def test_options_apply_only_to_their_command(backend, command, option, message):
    with pytest.raises(ValueError, match=message):
        main.run(args(command, "x", **option))


def test_delete_removes_the_document(backend):
    main.run(args("delete", "pdf-1"))
    main.kb.delete_document.assert_called_once_with(backend.context, "pdf-1", None)
    main.storage.connect_storage.assert_called_once_with(main.functions, create=False)


def test_list_reads_only_the_docstore(backend, capsys):
    main.run(args("list"))
    assert "pdf-1 | leave.pdf | pdf | 4 chunks" in capsys.readouterr().out
    main.storage.connect_storage.assert_not_called()
    main.create_models.assert_not_called()


def test_import_catalog_needs_the_oncall_domain(backend):
    with pytest.raises(ValueError, match="KB_DOMAIN=oncall"):
        main.run(args("import-catalog", "service_catalog.yaml"))
    main.storage.connect_storage.assert_not_called()


def test_import_catalog_reads_both_files(oncall, tmp_path, capsys):
    (tmp_path / "catalog.yaml").write_text("services: []", encoding="utf-8")
    (tmp_path / "alerts.yaml").write_text("alerts: []", encoding="utf-8")
    imported = SimpleNamespace(components={"cart": 1}, teams={"t": 1}, alerts={})
    main.kb.import_catalog.return_value = (imported, 1)
    main.run(
        args("import-catalog", str(tmp_path / "catalog.yaml"), alerts=str(tmp_path / "alerts.yaml"))
    )
    call = main.kb.import_catalog.call_args.args
    assert call[:2] == ("services: []", "alerts: []")
    assert call[4] is oncall.graph_store
    main.storage.connect_storage.assert_called_once_with(main.functions, create=True)
    assert "Imported 1 components, 1 teams, 0 alerts" in capsys.readouterr().out


def test_markdown_add_passes_graph_and_reports_extraction(oncall, capsys):
    main.kb.add_document.return_value = (
        "md-1",
        2,
        {"chunks": 2, "empty_chunks": 1, "dropped": ["SERVICE:billing"], "relations": 4},
    )
    main.run(args("add", "INC-001.md", doc_type="postmortem"))
    kwargs = main.kb.add_document.call_args.kwargs
    assert kwargs["doc_type"] == "postmortem"
    assert kwargs["graph_store"] is oncall.graph_store and kwargs["catalog"] is oncall.catalog
    assert kwargs["extraction"] == ("extractor", "cache")
    assert "Graph: 4 relations" in capsys.readouterr().out


def test_oncall_delete_also_cleans_the_graph(oncall, backend):
    main.run(args("delete", "md-1"))
    main.kb.delete_document.assert_called_once_with(backend.context, "md-1", oncall.graph_store)


@pytest.mark.parametrize("mode", ["hybrid", "graph", "hybrid+graph"])
def test_oncall_ask_uses_the_requested_mode(oncall, mode, capsys):
    main.run(args("ask", "Who owns checkout?", mode=mode))
    builder = main.graph_retrieval.build_oncall_query_engine
    assert builder.call_args.args[3] == mode
    assert f"Timing ({mode})" in capsys.readouterr().out


def test_oncall_ask_falls_back_to_hybrid_without_a_catalog(oncall, monkeypatch, capsys):
    monkeypatch.setattr(main.kb, "connect_graph", Mock(return_value=(object(), None)))
    main.run(args("ask", "Who owns checkout?", mode="graph"))
    assert main.graph_retrieval.build_oncall_query_engine.call_args.args[3] == "hybrid"
    assert "Using hybrid" in capsys.readouterr().err


def test_oncall_ask_works_when_neo4j_is_down(oncall, monkeypatch, capsys):
    monkeypatch.setattr(main.kb, "connect_graph", Mock(side_effect=RuntimeError("refused")))
    main.run(args("ask", "Who owns checkout?"))
    assert main.graph_retrieval.build_oncall_query_engine.call_args.args[3] == "hybrid"
    assert "Graph retrieval unavailable (refused)" in capsys.readouterr().err


def test_oncall_writes_require_neo4j(oncall, monkeypatch):
    monkeypatch.setattr(main.kb, "connect_graph", Mock(side_effect=RuntimeError("refused")))
    with pytest.raises(RuntimeError, match="refused"):
        main.run(args("add", "runbook.md"))
    main.kb.add_document.assert_not_called()
