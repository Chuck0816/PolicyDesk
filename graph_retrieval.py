"""Graph retrieval for on-call questions and the hybrid / graph / hybrid+graph modes.

Entities are linked deterministically from the catalog alias table, the question's intent
picks fixed Cypher templates (the LLM never writes Cypher), and the results become:
  * graph facts: short "Graph path" lines kept in reserved slots after reranking, and
  * graph chunks: source passages behind those facts, fused with vector and BM25 by RRF.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from catalog import CATALOG_FILE_NAME
from graph_schema import contains_term, match_symptoms
from prompts import ONCALL_QA_PROMPT
from retrieval import (
    CANDIDATE_TOP_K,
    FUSION_TOP_K,
    QueryRewriteRetriever,
    build_bm25_retriever,
    build_policy_retriever,
    load_index_nodes,
)
from services import PolicyQueryEngine

MODES = ("hybrid", "graph", "hybrid+graph")
MAX_HOPS = 3
MAX_GRAPH_FACTS = 5
# The original question, so entity linking never depends on the rewritten query.
ORIGINAL_QUESTION = ContextVar("original_question", default="")

INTENT_TERMS = {
    "impact": (
        "影响",
        "波及",
        "会影响",
        "impact",
        "affect",
        "affected",
        "blast radius",
        "who uses",
    ),
    "root_cause": ("根因", "原因", "导致", "为什么", "root cause", "cause", "why", "caused"),
    "owner": ("负责", "团队", "找谁", "升级", "owner", "owns", "team", "escalat", "on-call"),
    "similar": ("以前", "之前", "类似", "历史", "发生过", "similar", "before", "previous", "past"),
}

# --- Cypher templates: parameters are bound by the driver; hop limits are validated ints. ---

IMPACT = """
MATCH p = (up:__Entity__)-[:DEPENDS_ON*1..{hops}]->(s:__Entity__ {{id: $service}})
WITH up, p ORDER BY length(p)
WITH up, collect(p)[0] AS shortest
RETURN up.id AS service, length(shortest) AS hops,
       [n IN nodes(shortest) | n.id] AS path,
       [r IN relationships(shortest) | r.type] AS dep_types
ORDER BY hops, service
"""

DOWNSTREAM = """
MATCH p = (s:__Entity__ {{id: $service}})-[:DEPENDS_ON*1..{hops}]->(down:__Entity__)
WITH down, p ORDER BY length(p)
WITH down, collect(p)[0] AS shortest
RETURN down.id AS service, length(shortest) AS hops,
       [n IN nodes(shortest) | n.id] AS path,
       [r IN relationships(shortest) | r.type] AS dep_types
ORDER BY hops, service
"""

OWNER = """
MATCH (s:__Entity__ {id: $service})-[:OWNED_BY]->(t:TEAM)
OPTIONAL MATCH p = (t)-[:ESCALATES_TO*1..3]->(:TEAM)
WITH t, p ORDER BY length(p) DESC
WITH t, collect(p)[0] AS longest
RETURN t.id AS team, coalesce([n IN nodes(longest) | n.id], [t.id]) AS escalation
"""

INCIDENTS = """
MATCH (i:INCIDENT)
WHERE EXISTS { (i)-[:SHOWED]->(sy:SYMPTOM) WHERE sy.id IN $symptoms }
   OR EXISTS { (i)-[:ROOT_CAUSE_IN|AFFECTED]->(c:__Entity__) WHERE c.id IN $services }
OPTIONAL MATCH (i)-[sr:SHOWED]->(sy:SYMPTOM)
WITH i, collect(DISTINCT {symptom: sy.id, service: sr.service}) AS symptoms,
     apoc.coll.flatten(collect(coalesce(sr.chunk_ids, []))) AS c1
OPTIONAL MATCH (i)-[rr:ROOT_CAUSE_IN]->(rc:__Entity__)
WITH i, symptoms, c1, collect(DISTINCT rc.id) AS root_causes,
     apoc.coll.flatten(collect(coalesce(rr.chunk_ids, []))) AS c2
OPTIONAL MATCH (i)-[mr:RESOLVED_BY]->(m:MITIGATION)
WITH i, symptoms, root_causes, c1 + c2 AS c12, collect(DISTINCT m.id) AS mitigations,
     apoc.coll.flatten(collect(coalesce(mr.chunk_ids, []))) AS c3
RETURN i.id AS incident, i.file_name AS file_name, symptoms, root_causes, mitigations,
       apoc.coll.toSet(c12 + c3) AS chunk_ids
"""

ALERTS = """
MATCH (a:ALERT)-[:FIRES_ON]->(s:__Entity__) WHERE a.id IN $alerts
OPTIONAL MATCH (a)-[:HANDLED_BY]->(rs:RUNBOOK_SECTION)
RETURN a.id AS alert, a.metric AS metric, s.id AS service,
       rs.file_name AS file_name, rs.section AS section
"""


def _hops(value):
    hops = int(value)
    if not 1 <= hops <= MAX_HOPS:
        raise ValueError(f"Hop limit must be between 1 and {MAX_HOPS}.")
    return hops


def detect_intents(text):
    return {
        intent
        for intent, words in INTENT_TERMS.items()
        if any(contains_term(text, w) for w in words)
    }


def _path_text(path, dep_types):
    parts = [path[0]]
    for node, dep_type in zip(path[1:], dep_types):
        parts.append(f"-[DEPENDS_ON {dep_type}]-> {node}")
    return " ".join(parts)


@dataclass
class GraphFact:
    kind: str  # impact | root_cause | owner | incident | alert
    text: str
    sources: list = field(default_factory=list)  # [(file_name, section)]
    chunk_ids: list = field(default_factory=list)
    catalog_services: list = field(default_factory=list)


@dataclass
class GraphResult:
    services: set
    symptoms: set
    alerts: set
    intents: set
    facts: list


class GraphSearch:
    """Link entities, run the templates the question needs, and rank root causes."""

    def __init__(self, graph_store, catalog, hops=MAX_HOPS):
        self.graph_store = graph_store
        self.catalog = catalog
        self.hops = _hops(hops)

    def _query(self, template, **params):
        return self.graph_store.structured_query(template, param_map=params)

    def impact(self, service):
        rows = self._query(IMPACT.format(hops=self.hops), service=service)
        if not rows:
            return [
                GraphFact(
                    "impact",
                    f"Graph: nothing depends on {service}.",
                    [(CATALOG_FILE_NAME, service)],
                    catalog_services=[service],
                )
            ]
        affected = ", ".join(f"{row['service']} ({row['hops']} hop)" for row in rows)
        paths = "; ".join(_path_text(row["path"], row["dep_types"]) for row in rows[:4])
        return [
            GraphFact(
                "impact",
                f"Graph path (impact of {service}): affected upstream services: {affected}. "
                f"Paths: {paths}.",
                [(CATALOG_FILE_NAME, service)],
                catalog_services=[service] + [row["service"] for row in rows],
            )
        ]

    def owner(self, service):
        facts = []
        for row in self._query(OWNER, service=service):
            chain = " -[ESCALATES_TO]-> ".join(row["escalation"])
            facts.append(
                GraphFact(
                    "owner",
                    f"Graph path (owner of {service}): {service} -[OWNED_BY]-> {chain}.",
                    [(CATALOG_FILE_NAME, service)],
                    catalog_services=[service],
                )
            )
        return facts

    def incidents(self, services, symptoms, limit=3):
        rows = self._query(INCIDENTS, services=sorted(services), symptoms=sorted(symptoms))

        def relevance(row):
            shared = {item["symptom"] for item in row["symptoms"]} & symptoms
            touched = set(row["root_causes"]) & services
            return (len(shared) + len(touched), row["incident"])

        rows.sort(key=relevance, reverse=True)
        facts = []
        for row in rows[:limit]:
            observed = (
                ", ".join(
                    f"{item['symptom']}" + (f" on {item['service']}" if item["service"] else "")
                    for item in row["symptoms"]
                    if item["symptom"]
                )
                or "not recorded"
            )
            facts.append(
                GraphFact(
                    "incident",
                    f"Graph path (past incident {row['incident']}): showed {observed}; "
                    f"root cause in {', '.join(row['root_causes']) or 'unknown'}; "
                    f"resolved by: {'; '.join(row['mitigations']) or 'not recorded'}.",
                    [(row["file_name"], row["incident"])],
                    chunk_ids=list(row["chunk_ids"]),
                )
            )
        return facts, rows

    def root_causes(self, services, symptoms, limit=3):
        """Rank candidates: shared downstream of all symptomatic services, then history."""
        reach = {}  # candidate -> {symptomatic service: (hops, path, dep_types)}
        for service in services:
            reach.setdefault(service, {})[service] = (0, [service], [])
            for row in self._query(DOWNSTREAM.format(hops=self.hops), service=service):
                reach.setdefault(row["service"], {})[service] = (
                    row["hops"],
                    row["path"],
                    row["dep_types"],
                )
        _, incident_rows = self.incidents(set(), symptoms, limit=50)
        history = {}
        for row in incident_rows:
            for cause in row["root_causes"]:
                history.setdefault(cause, []).append(row["incident"])

        scored = []
        for candidate, sources in reach.items():
            coverage = len(sources) / max(len(services), 1)
            distance = sum(hops for hops, _, _ in sources.values()) / len(sources)
            past = history.get(candidate, [])
            score = 2.0 * coverage + 1.0 / (1.0 + distance) + 0.5 * min(len(past), 3)
            scored.append((score, candidate, sources, past))
        scored.sort(key=lambda item: (-item[0], item[1]))

        facts = []
        for score, candidate, sources, past in scored[:limit]:
            reasons = [
                "reported symptom here"
                if hops == 0
                else f"downstream of {origin} ({hops} hop): {_path_text(path, dep_types)}"
                for origin, (hops, path, dep_types) in sources.items()
            ]
            if past:
                reasons.append(f"root cause of past incidents {', '.join(sorted(past))}")
            facts.append(
                GraphFact(
                    "root_cause",
                    f"Graph reasoning (root-cause candidate {candidate}, score {score:.2f}): "
                    + "; ".join(reasons)
                    + ".",
                    [(CATALOG_FILE_NAME, candidate)],
                    catalog_services=[candidate],
                )
            )
        return facts

    def alerts(self, alerts):
        facts = []
        for row in self._query(ALERTS, alerts=sorted(alerts)):
            runbook = (
                f" -[HANDLED_BY]-> runbook {row['file_name']} § {row['section']}"
                if row["file_name"]
                else ""
            )
            facts.append(
                GraphFact(
                    "alert",
                    f"Graph path: alert {row['alert']} ({row['metric'] or 'metric not set'}) "
                    f"-[FIRES_ON]-> {row['service']}{runbook}.",
                    [(row["file_name"], row["section"])] if row["file_name"] else [],
                    catalog_services=[row["service"]],
                )
            )
        return facts

    def search(self, *texts):
        text = "\n".join(t for t in texts if t)
        alerts = self.catalog.find_alerts(text)
        services = self.catalog.find_components(text)
        services |= {self.catalog.alerts[a]["service"] for a in alerts}
        symptoms = match_symptoms(text)
        intents = detect_intents(text)

        facts = []
        if alerts:
            facts += self.alerts(alerts)
        # Root causes are ranked when asked for, or when symptoms come with no other intent.
        if services and ("root_cause" in intents or (symptoms and not intents)):
            facts += self.root_causes(services, symptoms)
        if "impact" in intents:
            facts += [fact for s in sorted(services) for fact in self.impact(s)]
        if "owner" in intents:
            facts += [fact for s in sorted(services) for fact in self.owner(s)]
        if "similar" in intents or symptoms:
            facts += self.incidents(services, symptoms)[0]
        if services and not facts:
            facts += [fact for s in sorted(services) for fact in self.owner(s)]
        return GraphResult(services, symptoms, alerts, intents, facts)


class GraphChunkRetriever(BaseRetriever):
    """Third RRF branch: passages behind the graph facts, in fact order."""

    def __init__(self, graph_search, nodes, top_k=CANDIDATE_TOP_K):
        super().__init__()
        self._search = graph_search
        self._top_k = top_k
        self._by_id = {node.node_id: node for node in nodes}
        self._by_doc, self._by_section = {}, {}
        for node in nodes:
            metadata = node.metadata
            self._by_doc.setdefault(metadata.get("source_doc_id"), []).append(node)
            key = (metadata.get("file_name"), metadata.get("section"))
            self._by_section.setdefault(key, []).append(node)

    def _retrieve(self, query_bundle):
        result = self._search.search(ORIGINAL_QUESTION.get(), query_bundle.query_str)
        ordered = []
        for fact in result.facts:
            ordered += [self._by_id[c] for c in fact.chunk_ids if c in self._by_id]
            for service in fact.catalog_services:
                ordered += self._by_doc.get(f"catalog-{service}", [])
            for file_name, section in fact.sources:
                if file_name != CATALOG_FILE_NAME:
                    ordered += [
                        node
                        for (f, s), group in self._by_section.items()
                        if f == file_name and s and section and section in s
                        for node in group
                    ]
        unique = list({node.node_id: node for node in ordered}.values())[: self._top_k]
        # Descending synthetic scores keep the graph's order inside RRF.
        return [
            NodeWithScore(node=node, score=1.0 / (1 + rank)) for rank, node in enumerate(unique)
        ]


class GraphFactsPostprocessor(BaseNodePostprocessor):
    """Append graph facts after reranking so the reranker cannot silently drop them."""

    max_facts: int = MAX_GRAPH_FACTS
    _graph_search: Any = PrivateAttr()

    def __init__(self, graph_search, max_facts=MAX_GRAPH_FACTS):
        super().__init__(max_facts=max_facts)
        self._graph_search = graph_search

    def _postprocess_nodes(self, nodes, query_bundle=None):
        question = query_bundle.query_str if query_bundle else ""
        facts = self._graph_search.search(question).facts[: self.max_facts]
        extra = [
            NodeWithScore(
                node=TextNode(
                    text=fact.text,
                    metadata={
                        "file_name": fact.sources[0][0] if fact.sources else CATALOG_FILE_NAME,
                        "section": fact.sources[0][1] if fact.sources else "",
                        "content_type": "graph_path",
                        "graph_kind": fact.kind,
                    },
                ),
                score=None,
            )
            for fact in facts
        ]
        return list(nodes) + extra


class OriginalQuestionRewriteRetriever(QueryRewriteRetriever):
    """Existing rewrite-and-fallback behavior, exposing the original question to the graph."""

    def _retrieve(self, query_bundle: QueryBundle):
        token = ORIGINAL_QUESTION.set(query_bundle.query_str)
        try:
            return super()._retrieve(query_bundle)
        finally:
            ORIGINAL_QUESTION.reset(token)


def resolve_mode(mode, graph_ready):
    if mode not in MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}; choose from {MODES}.")
    return mode if graph_ready else "hybrid"


def build_retriever(index, llm, mode, graph_search=None):
    """hybrid is exactly the existing PolicyDesk chain; the other modes add the graph."""
    if mode == "hybrid" or graph_search is None:
        return build_policy_retriever(index, llm)
    nodes = load_index_nodes(index)
    candidates = min(CANDIDATE_TOP_K, len(nodes))
    branches = [GraphChunkRetriever(graph_search, nodes, top_k=candidates)]
    if mode == "hybrid+graph":
        branches = [
            index.as_retriever(similarity_top_k=candidates),
            build_bm25_retriever(nodes, top_k=candidates),
            *branches,
        ]
    fusion = QueryFusionRetriever(
        retrievers=branches,
        llm=llm,
        mode="reciprocal_rerank",
        similarity_top_k=min(FUSION_TOP_K, len(nodes)),
        num_queries=1,
        use_async=False,
    )
    return OriginalQuestionRewriteRetriever(fusion, llm)


def build_oncall_query_engine(index, llm, reranker, mode, graph_search=None):
    """Streaming engine for one mode. Graph facts are appended only in graph modes."""
    retriever = build_retriever(index, llm, mode, graph_search)
    postprocessors = [reranker]
    if mode != "hybrid" and graph_search is not None:
        postprocessors.append(GraphFactsPostprocessor(graph_search))
    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=ResponseMode.COMPACT,
        streaming=True,
        text_qa_template=ONCALL_QA_PROMPT,
    )
    return PolicyQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=postprocessors,
    )
