"""Collection lifecycle checks without connecting to database services."""

from unittest.mock import Mock

import pytest

import storage


@pytest.mark.parametrize("create", [False, True])
def test_only_build_can_create_a_chroma_collection(monkeypatch, create):
    client = Mock()
    monkeypatch.setattr("storage.MongoDocumentStore.from_uri", Mock())
    monkeypatch.setattr("storage.RedisIndexStore.from_host_and_port", Mock())
    monkeypatch.setattr("storage.chromadb.HttpClient", Mock(return_value=client))
    monkeypatch.setattr("storage.ChromaVectorStore", Mock())
    storage.connect_storage(Mock(), create=create)
    expected = client.get_or_create_collection if create else client.get_collection
    unexpected = client.get_collection if create else client.get_or_create_collection
    expected.assert_called_once_with(storage.KB_NAME)
    unexpected.assert_not_called()
