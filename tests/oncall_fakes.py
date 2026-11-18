"""Shared on-call test doubles: the fixture catalog and a fake schema extractor."""

from pathlib import Path

from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    EntityNode,
    Relation,
)
from llama_index.core.llms import MockLLM
from llama_index.core.schema import TransformComponent

import catalog as service_catalog

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "oncall"
POSTMORTEMS = FIXTURES / "postmortems"
RUNBOOKS = FIXTURES / "runbooks"

# What an LLM might return: raw names, aliases, and noise the aligner must fix or drop.
RAW_TRIPLETS = {
    "INC-001": [
        (
            "the outage",
            "INCIDENT",
            "SHOWED",
            "error rate high",
            "SYMPTOM",
            {"service": "checkoutservice"},
        ),  # noqa: E501
        ("the outage", "INCIDENT", "ROOT_CAUSE_IN", "PaymentService", "SERVICE", {}),
        ("the outage", "INCIDENT", "AFFECTED", "Checkout", "SERVICE", {}),
        ("the outage", "INCIDENT", "RESOLVED_BY", "Roll back payment to v2.3", "MITIGATION", {}),
        ("the outage", "INCIDENT", "AFFECTED", "billing-ledger", "SERVICE", {}),
    ],
    "INC-002": [
        (
            "kafka lag",
            "INCIDENT",
            "SHOWED",
            "consumer lag",
            "SYMPTOM",
            {"service": "fraud detection"},
        ),  # noqa: E501
        ("kafka lag", "INCIDENT", "SHOWED", "errors", "SYMPTOM", {"service": "checkout"}),
        ("kafka lag", "INCIDENT", "ROOT_CAUSE_IN", "fraud-detection", "SERVICE", {}),
        ("kafka lag", "INCIDENT", "AFFECTED", "Kafka", "SERVICE", {}),
        (
            "kafka lag",
            "INCIDENT",
            "RESOLVED_BY",
            "Scale fraud-detection consumers",
            "MITIGATION",
            {},
        ),  # noqa: E501
    ],
}


def load_fixture_catalog():
    return service_catalog.load_catalog(FIXTURES / "service_catalog.yaml")


class FakeExtractor(TransformComponent):
    """Behaves like SchemaLLMPathExtractor: adds KG nodes and relations to chunk metadata."""

    def __init__(self, fail=False):
        super().__init__()
        self.__dict__["llm"] = MockLLM()
        self.__dict__["calls"] = 0
        self.__dict__["fail"] = fail

    def __call__(self, nodes, **kwargs):
        self.__dict__["calls"] += 1
        if self.fail:
            raise RuntimeError("extraction quota exceeded")
        for node in nodes:
            kg_nodes, kg_relations = [], []
            if "Summary" in node.metadata.get("section", ""):
                incident = node.metadata["incident_id"]
                for source, s_label, label, target, t_label, props in RAW_TRIPLETS[incident]:
                    a = EntityNode(name=source, label=s_label)
                    b = EntityNode(name=target, label=t_label)
                    kg_nodes += [a, b]
                    kg_relations.append(
                        Relation(label=label, source_id=a.id, target_id=b.id, properties=props)
                    )
            node.metadata[KG_NODES_KEY] = kg_nodes
            node.metadata[KG_RELATIONS_KEY] = kg_relations
        return nodes
