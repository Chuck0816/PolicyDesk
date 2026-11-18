"""Service catalog parsing, alias linking, validation, graph import, and rendered text."""

import pytest
from oncall_fakes import FIXTURES, load_fixture_catalog

import catalog as service_catalog
from graph_schema import canonical_symptom, contains_term, match_symptoms


class RecordingStore:
    """Records Cypher and upserts; answers the catalog-source query from what was saved."""

    def __init__(self):
        self.queries, self.nodes, self.relations, self.saved = [], [], [], None

    def structured_query(self, query, param_map=None):
        self.queries.append((query, param_map or {}))
        if query.startswith("MERGE (c:CatalogSource"):
            self.saved = param_map
        if query.startswith("MATCH (c:CatalogSource"):
            return [self.saved] if self.saved else []
        return []

    def upsert_nodes(self, nodes):
        self.nodes += nodes

    def upsert_relations(self, relations):
        self.relations += relations


@pytest.fixture
def catalog():
    return load_fixture_catalog()


def test_aliases_resolve_generated_and_declared_names(catalog):
    assert catalog.resolve("paymentservice") == "payment"
    assert catalog.resolve("Product Catalog") == "product-catalog"
    assert catalog.resolve("product_catalog") == "product-catalog"
    assert catalog.resolve("支付服务") == "payment"
    assert catalog.resolve("billing-ledger") is None


def test_find_components_in_mixed_language_text(catalog):
    found = catalog.find_components("支付服务挂了会影响谁？checkout 也慢，product-catalog 呢")
    assert found == {"payment", "checkout", "product-catalog"}
    # The longest alias wins: "valkey-cart" is not also "cart".
    assert catalog.find_components("谁负责 valkey-cart？") == {"valkey-cart"}
    assert catalog.find_components("cart 和 valkey-cart") == {"cart", "valkey-cart"}
    # Whole words only: "cart" must not match inside "carton".
    assert catalog.find_components("the carton shipment") == set()
    assert catalog.find_alerts("CartCacheConnectionsHigh is firing") == {"CartCacheConnectionsHigh"}


def test_escalation_and_upstream(catalog):
    assert catalog.escalation_path("team-checkout") == ["team-checkout", "team-sre"]
    assert sorted(catalog.upstream_of("cart")) == [("checkout", "rpc"), ("frontend", "rpc")]
    assert catalog.components["valkey-cart"].kind == "DATASTORE"


BASE = """
teams: [{name: t1}]
services:
  - {name: a, team: t1}
"""


@pytest.mark.parametrize(
    "text, alerts, message",
    [
        ("services: [", "", "not valid YAML"),
        ("teams: [{name: t1}]", "", "at least one service"),
        ("teams: [{name: t1}]\nservices: [{name: a}]", "", "missing a required field"),
        (BASE + "  - {name: a, team: t1}\n", "", "defined twice"),
        ("teams: [{name: a}]\nservices: [{name: a, team: a}]", "", "unique"),
        ("teams: [{name: t1}]\nservices: [{name: a, team: t2}]", "", "unknown team"),
        (
            BASE.replace("team: t1}", "team: t1, depends_on: [{target: b, type: rpc}]}"),
            "",
            "invalid target",
        ),  # noqa: E501
        (
            "teams: [{name: t1}]\nservices:\n  - {name: a, team: t1}\n"
            "  - {name: b, team: t1, depends_on: [{target: a, type: grpc}]}\n",
            "",
            "unknown type",
        ),
        (BASE, "alerts: [{name: X, service: nope}]", "unknown service"),
        (
            "teams: [{name: t1}]\nservices:\n  - {name: a, team: t1, aliases: [shop]}\n"
            "  - {name: b, team: t1, aliases: [shop]}\n",
            "",
            "Alias",
        ),
        ("teams: [{name: t1, escalates_to: t9}]\nservices: [{name: a, team: t1}]", "", "escalates"),
    ],
)
def test_invalid_catalogs_are_rejected(text, alerts, message):
    with pytest.raises(ValueError, match=message):
        service_catalog.parse_catalog(text, alerts)


def test_alerts_file_is_merged(catalog):
    text = (FIXTURES / "service_catalog.yaml").read_text(encoding="utf-8")
    merged = service_catalog.parse_catalog(text, "alerts: [{name: PaymentDown, service: payment}]")
    assert set(merged.alerts) == {"CartCacheConnectionsHigh", "PaymentDown"}


def test_catalog_graph_tags_every_fact_with_the_catalog_source(catalog):
    nodes, relations = service_catalog.catalog_graph(catalog)
    labels = {node.name: node.label for node in nodes}
    assert labels["checkout"] == "SERVICE" and labels["kafka"] == "DATASTORE"
    assert labels["team-sre"] == "TEAM" and labels["CartCacheConnectionsHigh"] == "ALERT"
    assert labels["runbook:cart.md#Cache connection pool exhausted"] == "RUNBOOK_SECTION"
    assert all(item.properties["source"] == "catalog" for item in nodes + relations)
    kinds = {relation.label for relation in relations}
    assert kinds == {"DEPENDS_ON", "OWNED_BY", "ESCALATES_TO", "FIRES_ON", "HANDLED_BY"}
    cache = next(r for r in relations if (r.source_id, r.target_id) == ("cart", "valkey-cart"))
    assert cache.properties["type"] == "cache"


def test_import_replaces_catalog_edges_and_keeps_document_facts(catalog):
    store = RecordingStore()
    assert service_catalog.import_catalog(store, catalog) == (
        len(store.nodes),
        len(store.relations),
    )
    delete_edges, delete_nodes = store.queries
    assert "r.source = $source DELETE r" in delete_edges[0]
    # Removed catalog nodes are deleted only when no document still references them.
    assert "size(coalesce(n.doc_ids, [])) = 0" in delete_nodes[0]
    assert "checkout" in delete_nodes[1]["keep"]


def test_catalog_source_round_trips_through_the_graph(catalog):
    store = RecordingStore()
    assert service_catalog.load_catalog_from_graph(store) is None
    text = (FIXTURES / "service_catalog.yaml").read_text(encoding="utf-8")
    service_catalog.save_catalog_source(store, text, "")
    loaded = service_catalog.load_catalog_from_graph(store)
    assert set(loaded.components) == set(catalog.components)


def test_catalog_documents_describe_each_component(catalog):
    documents = service_catalog.catalog_documents(catalog, "2026-11-18T00:00:00+00:00")
    assert len(documents) == len(catalog.components)
    cart = next(d for d in documents if d.id_ == "catalog-cart")
    assert "Owner team: team-checkout. Escalation path: team-checkout -> team-sre." in cart.text
    assert "Depends on: valkey-cart (cache)." in cart.text
    assert "Used by: frontend (rpc), checkout (rpc)." in cart.text
    assert "Alerts: CartCacheConnectionsHigh." in cart.text
    assert cart.metadata["doc_type"] == "service_catalog"
    assert "uploaded_at" in cart.excluded_embed_metadata_keys


def test_symptom_vocabulary():
    found = match_symptoms("checkout P99 延迟飙升，错误率上升")
    assert found == {"latency_high", "error_rate_high"}
    assert canonical_symptom("consumer lag") == "consumer_lag"
    assert canonical_symptom("Error Rate High") == "error_rate_high"
    assert canonical_symptom("disk full") is None
    assert not contains_term("download the logs", "down")
    assert contains_term("服务宕机了", "宕机")
