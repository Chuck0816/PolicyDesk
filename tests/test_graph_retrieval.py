"""Entity linking, Cypher templates, root-cause ranking, and the three retrieval modes."""

from collections import deque
from uuid import uuid4

import chromadb
import pytest
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from oncall_fakes import load_fixture_catalog

import catalog as service_catalog
import graph_retrieval
import services
import storage

INCIDENT_ROWS = [
    {
        "incident": "INC-001",
        "file_name": "INC-001.md",
        "symptoms": [{"symptom": "error_rate_high", "service": "checkout"}],
        "root_causes": ["payment"],
        "mitigations": ["Roll back payment to v2.3"],
        "chunk_ids": ["inc1-summary"],
    },
    {
        "incident": "INC-002",
        "file_name": "INC-002.md",
        "symptoms": [
            {"symptom": "consumer_lag", "service": "fraud-detection"},
            {"symptom": "error_rate_high", "service": "checkout"},
        ],
        "root_causes": ["fraud-detection"],
        "mitigations": ["Scale fraud-detection consumers"],
        "chunk_ids": ["inc2-summary"],
    },
]


class CatalogGraph:
    """Answers each Cypher template from the fixture catalog, like Neo4j would."""

    def __init__(self, catalog):
        self.catalog = catalog
        self.queries = []

    def _walk(self, start, edges, hops):
        rows, seen, queue = [], {start}, deque([(start, [start], [])])
        while queue:
            node, path, types = queue.popleft()
            if len(path) > hops:
                continue
            for nxt, dep_type in edges(node):
                if nxt not in seen:
                    seen.add(nxt)
                    row_path, row_types = path + [nxt], types + [dep_type]
                    rows.append(
                        {
                            "service": nxt,
                            "hops": len(row_types),
                            "path": row_path,
                            "dep_types": row_types,
                        }
                    )
                    queue.append((nxt, row_path, row_types))
        return sorted(rows, key=lambda row: (row["hops"], row["service"]))

    def structured_query(self, query, param_map=None):
        params = param_map or {}
        self.queries.append(query)
        components = self.catalog.components
        if "{id: $service})-[:DEPENDS_ON" in query:  # downstream
            return self._walk(params["service"], lambda n: components[n].depends_on, 3)
        if "-[:DEPENDS_ON*" in query:  # impact (upstream); paths read caller -> service
            rows = self._walk(params["service"], self.catalog.upstream_of, 3)
            for row in rows:
                row["path"].reverse()
                row["dep_types"].reverse()
            return rows
        if "OWNED_BY" in query:
            team = components[params["service"]].team
            return [{"team": team, "escalation": self.catalog.escalation_path(team)}]
        if "MATCH (i:INCIDENT)" in query:
            return [
                dict(row)
                for row in INCIDENT_ROWS
                if {s["symptom"] for s in row["symptoms"]} & set(params["symptoms"])
                or set(row["root_causes"]) & set(params["services"])
            ]
        if "MATCH (a:ALERT)" in query:
            return [
                {
                    "alert": name,
                    "metric": info["metric"],
                    "service": info["service"],
                    "file_name": info["runbook_file"],
                    "section": info["runbook_section"],
                }
                for name, info in self.catalog.alerts.items()
                if name in params["alerts"]
            ]
        raise AssertionError(f"Unexpected query: {query}")


@pytest.fixture
def search():
    catalog = load_fixture_catalog()
    return graph_retrieval.GraphSearch(CatalogGraph(catalog), catalog)


def kinds(result):
    return [fact.kind for fact in result.facts]


def test_intents_and_hop_limits():
    assert graph_retrieval.detect_intents("如果 cart 挂了会影响谁？该找哪个团队？") == {
        "impact",
        "owner",
    }
    assert graph_retrieval.detect_intents("What caused the outage?") == {"root_cause"}
    with pytest.raises(ValueError, match="Hop limit"):
        graph_retrieval.GraphSearch(None, None, hops=4)
    assert "*1..2]" in graph_retrieval.IMPACT.format(hops=2)


def test_impact_lists_upstream_services_with_paths(search):
    result = search.search("如果 product-catalog 不可用，会影响哪些服务？")
    assert result.services == {"product-catalog"} and kinds(result) == ["impact"]
    text = result.facts[0].text
    assert "checkout (1 hop)" in text and "frontend (1 hop)" in text
    assert "checkout -[DEPENDS_ON rpc]-> product-catalog" in text
    assert set(result.facts[0].catalog_services) == {"product-catalog", "checkout", "frontend"}


def test_owner_follows_the_escalation_chain(search):
    result = search.search("Who owns checkout and where do we escalate?")
    assert kinds(result) == ["owner"]
    assert "checkout -[OWNED_BY]-> team-checkout -[ESCALATES_TO]-> team-sre" in result.facts[0].text


def test_root_cause_prefers_the_shared_dependency_with_history(search):
    result = search.search("checkout P99 延迟飙升，同时 payment 错误率上升，最可能的根因是什么？")
    assert result.services == {"checkout", "payment"}
    assert result.symptoms == {"latency_high", "error_rate_high"}
    causes = [fact for fact in result.facts if fact.kind == "root_cause"]
    assert "root-cause candidate payment" in causes[0].text
    assert "root cause of past incidents INC-001" in causes[0].text
    assert "downstream of checkout (1 hop)" in causes[0].text
    # Incidents with the same symptoms are listed too, with their source chunks.
    incidents = [fact for fact in result.facts if fact.kind == "incident"]
    assert {fact.chunk_ids[0] for fact in incidents} == {"inc1-summary", "inc2-summary"}


def test_similar_question_lists_incidents_without_root_cause_ranking(search):
    result = search.search("以前有没有发生过类似的 Kafka 积压？当时怎么解决的？")
    assert result.intents == {"similar"} and result.symptoms == {"consumer_lag"}
    assert kinds(result) == ["incident"]
    assert "resolved by: Scale fraud-detection consumers" in result.facts[0].text


def test_symptoms_alone_trigger_root_cause_ranking(search):
    result = search.search("checkout 错误率很高")
    assert "root_cause" in kinds(result) and "incident" in kinds(result)


def test_alert_links_to_service_and_runbook_section(search):
    result = search.search("CartCacheConnectionsHigh is firing")
    assert "cart" in result.services
    alert = result.facts[0]
    assert alert.kind == "alert"
    assert "-[FIRES_ON]-> cart -[HANDLED_BY]-> runbook cart.md" in alert.text
    assert alert.sources == [("cart.md", "Cache connection pool exhausted")]


def test_a_service_without_intent_falls_back_to_its_owner(search):
    assert kinds(search.search("checkout 的 SLO 是多少？")) == ["owner"]
    assert search.search("What is the weather?").facts == []


def chunk(node_id, text, **metadata):
    return TextNode(id_=node_id, text=text, metadata={"content_type": "text", **metadata})


CHUNKS = [
    chunk("inc1-summary", "Payment rejected charges.", file_name="INC-001.md", section="Summary"),
    chunk(
        "cat-payment",
        "Service payment: charges cards.",
        file_name=service_catalog.CATALOG_FILE_NAME,
        section="payment",
        source_doc_id="catalog-payment",
    ),
    chunk(
        "rb-cart",
        "Restart cart pods.",
        file_name="cart.md",
        section="Cart runbook > Cache connection pool exhausted",
    ),
    chunk("other", "Unrelated text.", file_name="misc.md", section="Intro"),
]


def test_graph_chunks_follow_facts_and_use_the_original_question(search):
    retriever = graph_retrieval.GraphChunkRetriever(search, CHUNKS)
    token = graph_retrieval.ORIGINAL_QUESTION.set("payment 错误率上升的根因？")
    try:
        # The rewritten query names no service; linking still uses the original question.
        found = retriever.retrieve("why are requests failing")
    finally:
        graph_retrieval.ORIGINAL_QUESTION.reset(token)
    ids = [item.node.node_id for item in found]
    assert ids[:2] == ["cat-payment", "inc1-summary"] and "other" not in ids
    assert [item.score for item in found] == sorted((item.score for item in found), reverse=True)
    alert = retriever.retrieve("CartCacheConnectionsHigh is firing")
    assert "rb-cart" in [item.node.node_id for item in alert]


def test_graph_facts_are_appended_after_reranking(search):
    processor = graph_retrieval.GraphFactsPostprocessor(search, max_facts=1)
    evidence = NodeWithScore(node=chunk("doc", "Evidence."), score=0.4)
    nodes = processor.postprocess_nodes(
        [evidence], query_bundle=QueryBundle("Who owns checkout and payment?")
    )
    assert nodes[0] is evidence and len(nodes) == 2
    fact = nodes[1]
    assert fact.node.metadata["content_type"] == "graph_path" and fact.score is None
    assert fact.node.metadata["section"] == "checkout"
    assert fact.node.get_content().startswith("Graph path (owner of checkout)")


def test_resolve_mode():
    assert graph_retrieval.resolve_mode("graph", True) == "graph"
    assert graph_retrieval.resolve_mode("hybrid+graph", False) == "hybrid"
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        graph_retrieval.resolve_mode("vector", True)


def test_hybrid_mode_is_exactly_the_existing_chain(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(graph_retrieval, "build_policy_retriever", lambda index, llm: sentinel)
    assert graph_retrieval.build_retriever("index", "llm", "hybrid", graph_search=object()) is (
        sentinel
    )
    assert graph_retrieval.build_retriever("index", "llm", "graph", graph_search=None) is sentinel


class KeepFirst(BaseNodePostprocessor):
    def _postprocess_nodes(self, nodes, query_bundle=None):
        return nodes[:3]


@pytest.mark.parametrize("mode", graph_retrieval.MODES)
def test_engines_return_evidence_and_graph_paths_per_mode(search, mode, monkeypatch):
    Settings.embed_model = MockEmbedding(embed_dim=8)
    llm = MockLLM(max_tokens=8)
    monkeypatch.setattr(graph_retrieval.QueryRewriteRetriever, "_predict_rewrite", lambda s, q: q)
    collection = chromadb.EphemeralClient().get_or_create_collection(f"g_{uuid4().hex}")
    context = services.build_storage_context(
        SimpleDocumentStore(), SimpleIndexStore(), ChromaVectorStore(chroma_collection=collection)
    )
    index = VectorStoreIndex(CHUNKS, storage_context=context, store_nodes_override=True)
    index.set_index_id(storage.INDEX_ID)
    engine = graph_retrieval.build_oncall_query_engine(index, llm, KeepFirst(), mode, search)
    response = engine.query("payment 错误率上升的根因是什么？")
    "".join(response.response_gen)
    kinds_seen = [node.metadata["content_type"] for node in response.source_nodes]
    documents = [kind for kind in kinds_seen if kind != "graph_path"]
    assert 0 < len(documents) <= 3
    if mode == "hybrid":
        assert "graph_path" not in kinds_seen
    else:
        assert kinds_seen[len(documents) :] == ["graph_path"] * (len(kinds_seen) - len(documents))
        assert len(kinds_seen) - len(documents) <= graph_retrieval.MAX_GRAPH_FACTS
        assert "graph_path" in kinds_seen
    if mode == "graph":
        # Graph-only retrieval returns just the passages behind the facts.
        ids = {node.node.node_id for node in response.source_nodes if node.score is not None}
        assert "other" not in ids
