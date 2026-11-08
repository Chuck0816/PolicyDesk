"""CLI scope propagation and conversation-key isolation without external services."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main


@pytest.fixture
def backend(monkeypatch):
    context = SimpleNamespace(index_store=Mock())
    monkeypatch.setattr(main.storage, "connect_storage", Mock(return_value=(context, Mock())))
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
    return SimpleNamespace(builder=builder, memory=memory, engine=engine)


@pytest.mark.parametrize("category", [None, "salary", "leave"])
def test_chat_keeps_each_scope_in_a_separate_history(backend, category):
    main.run(Namespace(command="chat", category=category, question=None, session="employee"))
    expected = f"{main.storage.KB_NAME}:chat:employee"
    if category is not None:
        expected += f":category:{category}"
    assert backend.memory.call_args.args[1] == expected
    assert backend.builder.call_args.kwargs["category"] == category
    assert backend.builder.call_args.kwargs["advanced"] is True


def test_ask_passes_scope_and_preserves_question(backend):
    backend.engine.query.return_value.source_nodes = []
    main.run(
        Namespace(
            command="ask", category="salary", question="When is payroll paid?", session="demo"
        )
    )
    backend.engine.query.assert_called_once_with("When is payroll paid?")
    assert backend.builder.call_args.kwargs["category"] == "salary"


@pytest.mark.parametrize("command", ["build", "preview"])
def test_category_cannot_accidentally_build_a_partial_index(backend, command):
    with pytest.raises(ValueError, match="only to ask and chat"):
        main.run(Namespace(command=command, category="salary", question=None, session="demo"))
    main.storage.connect_storage.assert_not_called()
