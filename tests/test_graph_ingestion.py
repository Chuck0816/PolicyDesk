"""Markdown chunking, extraction alignment and caching, graph writes, and rollback."""

from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import chromadb
import pytest
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.kvstore import SimpleKVStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from oncall_fakes import POSTMORTEMS, RUNBOOKS, FakeExtractor, load_fixture_catalog

import graph_ingestion
import services


class RecordingGraphStore(SimplePropertyGraphStore):
    """In-memory property graph that records the provenance and cleanup Cypher."""

    def structured_query(self, query, param_map=None):
        self.__dict__.setdefault("queries", []).append((" ".join(query.split()), param_map))
        if "RETURN c.id AS id" in query:
            return [{"id": node_id} for node_id in self.__dict__.get("chunk_ids", [])]
        return []


@pytest.fixture
def catalog():
    return load_fixture_catalog()


@pytest.fixture
def knowledge_base():
    Settings.embed_model = MockEmbedding(embed_dim=8)
    collection = chromadb.EphemeralClient().get_or_create_collection(f"kb_{uuid4().hex}")
    context = services.build_storage_context(
        SimpleDocumentStore(), SimpleIndexStore(), ChromaVectorStore(chroma_collection=collection)
    )
    return context, collection


def test_front_matter_and_fence_aware_sections():
    front, body = graph_ingestion.parse_front_matter((RUNBOOKS / "cart.md").read_text())
    assert front == {"service": "cart"}
    sections = graph_ingestion.split_sections(body)
    assert [title for title, _ in sections] == [
        "Cart runbook",
        "Cart runbook > Cache connection pool exhausted",
    ]
    # The "# not a heading" line inside the code fence stays in its section.
    assert "# not a heading inside code" in sections[1][1]
    assert graph_ingestion.parse_front_matter("# Title\ntext") == ({}, "# Title\ntext")


def test_markdown_nodes_carry_metadata_and_one_source_document():
    text = (POSTMORTEMS / "INC-001.md").read_text()
    nodes = graph_ingestion.markdown_nodes(
        text, file_name="INC-001.md", doc_type="postmortem", doc_id="md-x"
    )
    assert [node.metadata["section"] for node in nodes] == ["Summary", "Resolution"]
    assert {node.ref_doc_id for node in nodes} == {"md-x"}
    metadata = nodes[0].metadata
    assert metadata["incident_id"] == "INC-001" and metadata["title"] == "Payment charge failures"
    assert "uploaded_at" in nodes[0].excluded_embed_metadata_keys


@pytest.mark.parametrize(
    "text, doc_type, message",
    [
        ("# Summary\ntext", "postmortem", "incident_id"),
        ("---\nincident_id: X\n---\n", "postmortem", "no text"),
        ("# A\ntext", "faq", "doc_type"),
    ],
)
def test_markdown_nodes_reject_bad_input(text, doc_type, message):
    with pytest.raises(ValueError, match=message):
        graph_ingestion.markdown_nodes(text, file_name="x.md", doc_type=doc_type, doc_id="md-x")


def test_alignment_maps_aliases_and_drops_unknown_entities(catalog):
    incident = EntityNode(name="the outage", label="INCIDENT")
    payment = EntityNode(name="PaymentService", label="SERVICE")
    unknown = EntityNode(name="billing-ledger", label="SERVICE")
    symptom = EntityNode(name="p99 latency", label="SYMPTOM")
    fuzzy = EntityNode(name="weird feeling", label="SYMPTOM")
    relations = [
        Relation(label="ROOT_CAUSE_IN", source_id=incident.id, target_id=payment.id),
        Relation(label="AFFECTED", source_id=incident.id, target_id=unknown.id),
        Relation(
            label="SHOWED",
            source_id=incident.id,
            target_id=symptom.id,
            properties={"service": "checkoutservice"},
        ),
        Relation(
            label="SHOWED",
            source_id=incident.id,
            target_id=fuzzy.id,
            properties={"service": "nowhere"},
        ),
    ]
    nodes, aligned, dropped = graph_ingestion.align_chunk_graph(
        catalog,
        {"incident_id": "INC-009"},
        [incident, payment, unknown, symptom, fuzzy],
        relations,
    )
    assert {(n.label, n.name) for n in nodes} == {
        ("INCIDENT", "INC-009"),
        ("SERVICE", "payment"),
        ("SYMPTOM", "latency_high"),
    }
    assert [(r.label, r.source_id, r.target_id) for r in aligned] == [
        ("ROOT_CAUSE_IN", "INC-009", "payment"),
        ("SHOWED", "INC-009", "latency_high"),
    ]
    assert aligned[1].properties["service"] == "checkout"
    assert dropped == ["SERVICE:billing-ledger", "SYMPTOM:weird feeling"]


def postmortem_nodes(name="INC-001"):
    text = (POSTMORTEMS / f"{name}.md").read_text()
    return graph_ingestion.markdown_nodes(
        text, file_name=f"{name}.md", doc_type="postmortem", doc_id=f"md-{name}"
    )


def test_extraction_is_cached_by_chunk_text(catalog):
    extractor, cache = FakeExtractor(), SimpleKVStore()
    graphs, report = graph_ingestion.extract_incident_graph(
        postmortem_nodes(), extractor, catalog, cache
    )
    assert extractor.calls == 1
    assert report == {"chunks": 2, "empty_chunks": 1, "dropped": ["SERVICE:billing-ledger"]}
    summary = next(iter(graphs.values()))
    assert {r.label for r in summary[1]} == {"SHOWED", "ROOT_CAUSE_IN", "AFFECTED", "RESOLVED_BY"}

    again, _ = graph_ingestion.extract_incident_graph(postmortem_nodes(), extractor, catalog, cache)
    assert extractor.calls == 1
    first = sorted((r.label, r.target_id) for _, rels in graphs.values() for r in rels)
    assert sorted((r.label, r.target_id) for _, rels in again.values() for r in rels) == first


def test_write_incident_graph_inserts_facts_and_records_provenance(catalog):
    Settings.embed_model = MockEmbedding(embed_dim=8)
    store = RecordingGraphStore()
    nodes = postmortem_nodes()
    graphs, _ = graph_ingestion.extract_incident_graph(nodes, FakeExtractor(), catalog)
    assert graph_ingestion.write_incident_graph(store, nodes, graphs, "md-INC-001") == 4
    triplets = {(s.id, r.label, t.id) for s, r, t in store.graph.get_triplets()}
    assert ("INC-001", "ROOT_CAUSE_IN", "payment") in triplets
    assert ("INC-001", "SHOWED", "error_rate_high") in triplets
    # Only the aligned facts: no default extractor output and no chunk-to-document links.
    assert {label for _, label, _ in triplets} == {
        "SHOWED",
        "ROOT_CAUSE_IN",
        "AFFECTED",
        "RESOLVED_BY",
    }
    edges, entities = store.queries
    assert "r.chunk_ids = apoc.coll.toSet" in edges[0]
    assert {row["doc_id"] for row in edges[1]["rows"]} == {"md-INC-001"}
    assert {row["id"] for row in entities[1]["rows"]} >= {"INC-001", "payment", "checkout"}


def test_delete_keeps_catalog_entities_and_facts_other_documents_support():
    store = RecordingGraphStore()
    store.__dict__["chunk_ids"] = ["c1", "c2"]
    assert graph_ingestion.delete_document_graph(store, "md-1") == 2
    queries = [query for query, _ in store.queries]
    assert "WITH r WHERE size(r.doc_ids) = 0 DELETE r" in queries[1]
    assert "e.doc_ids = [x IN e.doc_ids WHERE x <> $doc]" in queries[2]
    assert store.queries[3][1] == {"chunks": ["c1", "c2"]}
    assert "coalesce(e.source, '') <> $catalog" in queries[4]


def test_ingest_postmortem_writes_vector_and_graph_stores(knowledge_base, catalog, monkeypatch):
    context, collection = knowledge_base
    write = Mock(return_value=4)
    monkeypatch.setattr(graph_ingestion, "write_incident_graph", write)
    steps = []
    doc_id, count, report = graph_ingestion.ingest_markdown(
        POSTMORTEMS / "INC-001.md",
        context,
        collection,
        doc_type="postmortem",
        catalog=catalog,
        graph_store="store",
        extractor=FakeExtractor(),
        progress=steps.append,
    )
    assert steps == ["parse", "extract", "index"]
    assert count == 2 and report["relations"] == 4 and report["empty_chunks"] == 1
    assert collection.count() == 2
    info = context.docstore.get_ref_doc_info(doc_id)
    assert info.metadata["graph_relations"] == 4
    assert write.call_args.args[0] == "store" and write.call_args.args[3] == doc_id
    with pytest.raises(ValueError, match="already in the knowledge base"):
        graph_ingestion.ingest_markdown(
            POSTMORTEMS / "INC-001.md", context, collection, doc_type="runbook"
        )


def test_runbooks_skip_the_graph(knowledge_base):
    context, collection = knowledge_base
    doc_id, count, report = graph_ingestion.ingest_markdown(
        RUNBOOKS / "cart.md", context, collection, doc_type="runbook", graph_store="unused"
    )
    assert doc_id.startswith("md-") and count == 2 and report == {"chunks": 2}


def test_failed_extraction_leaves_every_store_unchanged(knowledge_base, catalog, monkeypatch):
    context, collection = knowledge_base
    write = Mock()
    monkeypatch.setattr(graph_ingestion, "write_incident_graph", write)
    with pytest.raises(RuntimeError, match="quota"):
        graph_ingestion.ingest_markdown(
            POSTMORTEMS / "INC-001.md",
            context,
            collection,
            doc_type="postmortem",
            catalog=catalog,
            graph_store="store",
            extractor=FakeExtractor(fail=True),
        )
    assert collection.count() == 0
    assert not context.docstore.get_all_ref_doc_info()
    write.assert_not_called()


def test_failed_graph_write_rolls_back_the_vector_insert(knowledge_base, catalog, monkeypatch):
    context, collection = knowledge_base
    monkeypatch.setattr(
        graph_ingestion, "write_incident_graph", Mock(side_effect=RuntimeError("neo4j down"))
    )
    cleanup = Mock()
    monkeypatch.setattr(graph_ingestion, "delete_document_graph", cleanup)
    with pytest.raises(RuntimeError, match="neo4j down"):
        graph_ingestion.ingest_markdown(
            POSTMORTEMS / "INC-001.md",
            context,
            collection,
            doc_type="postmortem",
            catalog=catalog,
            graph_store="store",
            extractor=FakeExtractor(),
        )
    assert collection.count() == 0
    assert not context.docstore.get_all_ref_doc_info()
    assert cleanup.call_args.args[0] == "store"


def test_postmortems_need_the_catalog_and_an_extractor(knowledge_base):
    context, collection = knowledge_base
    with pytest.raises(ValueError, match="catalog and an extractor"):
        graph_ingestion.ingest_markdown(
            POSTMORTEMS / "INC-002.md",
            context,
            collection,
            doc_type="postmortem",
            graph_store=SimpleNamespace(),
        )
