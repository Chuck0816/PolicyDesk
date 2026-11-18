"""End-to-end graph checks against a real Neo4j with APOC.

The catalog import and document cleanup run database-wide Cypher, and Neo4j Community has a
single database, so this test WIPES the graph. Run it only against a throwaway Neo4j:

    POLICYDESK_INTEGRATION=1 POLICYDESK_NEO4J_WIPE_OK=1 NEO4J_PASSWORD=... pytest tests/test_graph_database.py
"""  # noqa: E501

import os
from uuid import uuid4

import chromadb
import pytest
from llama_index.core import Settings, load_index_from_storage
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from oncall_fakes import POSTMORTEMS, RUNBOOKS, FakeExtractor, load_fixture_catalog

import catalog as service_catalog
import graph_ingestion
import graph_retrieval
import services
import storage

pytestmark = pytest.mark.skipif(
    not (
        os.getenv("POLICYDESK_INTEGRATION") == "1"
        and os.getenv("POLICYDESK_NEO4J_WIPE_OK") == "1"
        and os.getenv("NEO4J_PASSWORD")
    ),
    reason="Requires a throwaway Neo4j: POLICYDESK_INTEGRATION=1, POLICYDESK_NEO4J_WIPE_OK=1, "
    "and NEO4J_PASSWORD.",
)


def counts(store):
    return store.structured_query(
        "MATCH (n) WITH count(n) AS nodes OPTIONAL MATCH ()-[r]->() "
        "RETURN nodes, count(r) AS relations"
    )[0]


class KeepFirst(BaseNodePostprocessor):
    def _postprocess_nodes(self, nodes, query_bundle=None):
        return nodes[:4]


@pytest.fixture
def graph():
    store = storage.connect_graph_store()
    store.structured_query("MATCH (n) DETACH DELETE n")
    yield store
    store.structured_query("MATCH (n) DETACH DELETE n")


def test_catalog_import_ingest_search_and_delete(graph, monkeypatch):
    Settings.embed_model = MockEmbedding(embed_dim=8)
    catalog = load_fixture_catalog()
    text = (POSTMORTEMS.parent / "service_catalog.yaml").read_text(encoding="utf-8")

    service_catalog.import_catalog(graph, catalog)
    service_catalog.save_catalog_source(graph, text)
    catalog_only = counts(graph)
    # Importing again is idempotent, and the catalog can be rebuilt from Neo4j.
    service_catalog.import_catalog(graph, catalog)
    assert counts(graph) == catalog_only
    assert set(service_catalog.load_catalog_from_graph(graph).components) == set(catalog.components)

    collection = chromadb.EphemeralClient().get_or_create_collection(f"g_{uuid4().hex}")
    context = services.build_storage_context(
        SimpleDocumentStore(), SimpleIndexStore(), ChromaVectorStore(chroma_collection=collection)
    )
    graph_ingestion.append_to_vector_index(
        context, collection, services.split_documents(service_catalog.catalog_documents(catalog))
    )
    extractor, doc_ids = FakeExtractor(), {}
    for name in ("INC-001", "INC-002"):
        doc_ids[name], _, report = graph_ingestion.ingest_markdown(
            POSTMORTEMS / f"{name}.md",
            context,
            collection,
            doc_type="postmortem",
            catalog=catalog,
            graph_store=graph,
            extractor=extractor,
        )
        assert report["relations"] == (4 if name == "INC-001" else 5)
    graph_ingestion.ingest_markdown(RUNBOOKS / "cart.md", context, collection, doc_type="runbook")

    # Aliases were aligned: no duplicate service nodes, provenance on every extracted edge.
    services_in_graph = graph.structured_query("MATCH (s:SERVICE) RETURN collect(s.id) AS ids")
    assert sorted(services_in_graph[0]["ids"]) == sorted(
        name for name, c in catalog.components.items() if c.kind == "SERVICE"
    )
    edges = graph.structured_query(
        "MATCH (i:INCIDENT)-[r]->(x) RETURN i.id AS i, type(r) AS rel, x.id AS x, "
        "r.doc_ids AS docs, r.chunk_ids AS chunks"
    )
    assert all(len(edge["docs"]) == 1 and edge["chunks"] for edge in edges)
    shared = graph.structured_query(
        "MATCH (s:SYMPTOM {id: 'error_rate_high'}) RETURN s.doc_ids AS docs"
    )[0]["docs"]
    assert sorted(shared) == sorted(doc_ids.values())

    search = graph_retrieval.GraphSearch(graph, catalog)
    impact = search.search("如果 product-catalog 不可用，会影响哪些服务？").facts[0].text
    assert "checkout (1 hop)" in impact and "frontend (1 hop)" in impact
    owner = search.search("Who owns cart?").facts[0].text
    assert "cart -[OWNED_BY]-> team-checkout -[ESCALATES_TO]-> team-sre" in owner
    cause = search.search("checkout 延迟飙升，payment 错误率上升，根因是什么？")
    assert "root-cause candidate payment" in cause.facts[0].text
    assert "INC-001" in cause.facts[0].text
    similar = search.search("以前有没有发生过类似的 Kafka 积压？")
    assert similar.facts[0].text.startswith("Graph path (past incident INC-002)")
    alert = search.search("CartCacheConnectionsHigh is firing").facts[0].text
    assert "HANDLED_BY]-> runbook cart.md § Cache connection pool exhausted" in alert

    monkeypatch.setattr(graph_retrieval.QueryRewriteRetriever, "_predict_rewrite", lambda s, q: q)
    index = load_index_from_storage(context, index_id=storage.INDEX_ID)
    engine = graph_retrieval.build_oncall_query_engine(
        index, MockLLM(max_tokens=8), KeepFirst(), "hybrid+graph", search
    )
    response = engine.query("checkout 错误率上升，和以前的 payment 故障类似吗？")
    "".join(response.response_gen)
    kinds = [node.metadata["content_type"] for node in response.source_nodes]
    assert "graph_path" in kinds and kinds[0] != "graph_path"

    # Deleting one postmortem keeps the symptom INC-002 still mentions.
    graph_ingestion.delete_document_graph(graph, doc_ids["INC-001"])
    remaining = graph.structured_query("MATCH (i:INCIDENT) RETURN collect(i.id) AS ids")[0]
    assert remaining["ids"] == ["INC-002"]
    symptom = graph.structured_query(
        "MATCH (s:SYMPTOM {id: 'error_rate_high'}) RETURN s.doc_ids AS docs"
    )
    assert symptom[0]["docs"] == [doc_ids["INC-002"]]
    graph_ingestion.delete_document_graph(graph, doc_ids["INC-002"])
    assert counts(graph) == catalog_only
