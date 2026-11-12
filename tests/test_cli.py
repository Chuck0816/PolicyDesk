"""CLI document management, questions, and conversation keys without external services."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main

DOCUMENT = {
    "doc_id": "pdf-1",
    "file_name": "leave.pdf",
    "uploaded_at": "2026-11-12T08:30:00+00:00",
    "skipped_figures": 0,
    "chunks": 4,
}


def args(command, target=None, content_list=None, session="demo"):
    return Namespace(command=command, target=target, content_list=content_list, session=session)


@pytest.fixture
def backend(monkeypatch):
    context = SimpleNamespace(index_store=Mock())
    monkeypatch.setattr(main.storage, "connect_storage", Mock(return_value=(context, Mock())))
    monkeypatch.setattr(main.storage, "connect_docstore", Mock())
    monkeypatch.setattr(main.pdf_ingestion, "list_pdfs", Mock(return_value=[DOCUMENT]))
    monkeypatch.setattr(main.pdf_ingestion, "ingest_pdf", Mock(return_value=("pdf-2", 3, [])))
    monkeypatch.setattr(main.pdf_ingestion, "delete_pdf", Mock())
    monkeypatch.setattr(main, "create_models", Mock(return_value=(object(), object())))
    monkeypatch.setattr(main, "create_reranker", Mock(return_value=object()))
    monkeypatch.setattr(main, "load_index_from_storage", Mock())
    monkeypatch.setattr(main.functions, "configure_settings", Mock())
    engine = Mock()
    builder = Mock(return_value=engine)
    monkeypatch.setattr(main.functions, "build_query_engine", builder)
    memory = Mock()
    monkeypatch.setattr(main.functions, "build_chat_memory", memory)
    monkeypatch.setattr(main.storage, "connect_chat_store", Mock())
    chat = SimpleNamespace(chat_history=[])
    monkeypatch.setattr(main.CondenseQuestionChatEngine, "from_defaults", Mock(return_value=chat))
    monkeypatch.setattr("builtins.input", lambda _: "/exit")
    return SimpleNamespace(builder=builder, memory=memory, engine=engine, context=context)


def test_chat_keeps_one_history_per_session(backend):
    main.run(args("chat", session="employee"))
    assert backend.memory.call_args.args[1] == f"{main.storage.KB_NAME}:chat:employee"
    assert backend.builder.call_args.kwargs["advanced"] is True


def test_ask_preserves_question(backend):
    backend.engine.query.return_value.source_nodes = []
    main.run(args("ask", "When is payroll paid?"))
    backend.engine.query.assert_called_once_with("When is payroll paid?")
    main.storage.connect_storage.assert_called_once_with(main.functions, create=False)


@pytest.mark.parametrize("command", ["ask", "chat"])
def test_questions_need_an_uploaded_pdf_before_loading_models(backend, command):
    main.pdf_ingestion.list_pdfs.return_value = []
    with pytest.raises(ValueError, match="has no PDFs"):
        main.run(args(command, "Who approves leave?"))
    main.create_models.assert_not_called()
    main.storage.connect_storage.assert_not_called()


@pytest.mark.parametrize("command", ["add", "delete", "ask"])
def test_missing_target_is_rejected_before_connecting(backend, command):
    with pytest.raises(ValueError, match="Provide"):
        main.run(args(command, "  "))
    main.storage.connect_storage.assert_not_called()


def test_add_creates_storage_and_reports_skipped_figures(backend, capsys):
    main.pdf_ingestion.ingest_pdf.return_value = (
        "pdf-2",
        3,
        [{"page": 4, "caption": "图 1", "reason": "RuntimeError: quota"}],
    )
    main.run(args("add", "travel.pdf"))
    main.storage.connect_storage.assert_called_once_with(main.functions, create=True)
    assert main.pdf_ingestion.ingest_pdf.call_args.args[0] == "travel.pdf"
    output = capsys.readouterr().out
    assert "Added pdf-2: 3 chunks" in output
    assert "Skipped figure on page 4 (图 1): RuntimeError: quota" in output


def test_add_can_reuse_a_mineru_content_list(backend, tmp_path):
    content_list = tmp_path / "travel_content_list.json"
    main.run(args("add", "travel.pdf", content_list=str(content_list)))
    parse = main.pdf_ingestion.ingest_pdf.call_args.kwargs["parse"]
    assert parse("travel.pdf", tmp_path) == content_list


def test_content_list_applies_only_to_add(backend):
    with pytest.raises(ValueError, match="only to add"):
        main.run(args("ask", "Who approves leave?", content_list="x.json"))


def test_delete_removes_the_document(backend):
    main.run(args("delete", "pdf-1"))
    main.pdf_ingestion.delete_pdf.assert_called_once_with(backend.context, "pdf-1")
    main.storage.connect_storage.assert_called_once_with(main.functions, create=False)


def test_list_reads_only_the_docstore(backend, capsys):
    main.run(args("list"))
    assert "pdf-1 | leave.pdf | 4 chunks" in capsys.readouterr().out
    main.storage.connect_storage.assert_not_called()
    main.create_models.assert_not_called()
