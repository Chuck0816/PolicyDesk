"""Load the service catalog, import it into Neo4j without an LLM, and render it as text.

The catalog is the single source of truth for services, datastores, teams, dependencies,
and alerts. Entity names are graph IDs, so every name must be unique across all types.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from llama_index.core import Document
from llama_index.core.graph_stores.types import EntityNode, Relation

from graph_schema import CATALOG_SOURCE, DEPENDENCY_TYPES, contains_term

CATALOG_FILE_NAME = "service_catalog.yaml"


@dataclass
class Component:
    name: str
    kind: str  # "SERVICE" or "DATASTORE"
    team: str
    description: str = ""
    slo: str = ""
    aliases: list = field(default_factory=list)
    depends_on: list = field(default_factory=list)  # [(target, type)]


@dataclass
class Catalog:
    components: dict
    teams: dict  # name -> escalates_to (or None)
    alerts: dict  # name -> {"service", "metric", "runbook_file", "runbook_section"}
    alias_index: dict = field(default_factory=dict)  # alias -> canonical name

    def resolve(self, text):
        """Return the canonical component name for an exact alias, or None."""
        return self.alias_index.get(_normalize(text))

    def find_components(self, text):
        """Return canonical components mentioned anywhere in free text."""
        # Aliases are stored normalized ("product catalog"), so normalize the text the same way.
        normalized = _normalize(text)
        spans = [
            (start, start + len(alias), name)
            for alias, name in self.alias_index.items()
            if contains_term(normalized, alias)
            for start in _occurrences(normalized, alias)
        ]
        # "valkey-cart" also contains the alias "cart"; keep only the longest overlapping match.
        return {
            name
            for start, end, name in spans
            if not any(s <= start and end <= e and (e - s) > (end - start) for s, e, _ in spans)
        }

    def find_alerts(self, text):
        return {name for name in self.alerts if contains_term(text, name)}

    def upstream_of(self, name):
        """Direct callers of a component, for the rendered text documents."""
        return [
            (caller.name, dep_type)
            for caller in self.components.values()
            for target, dep_type in caller.depends_on
            if target == name
        ]

    def escalation_path(self, team):
        path, seen = [team], {team}
        while (nxt := self.teams.get(path[-1])) and nxt not in seen:
            path.append(nxt)
            seen.add(nxt)
        return path


def _occurrences(text, term):
    start = text.find(term)
    while start != -1:
        yield start
        start = text.find(term, start + 1)


def _normalize(text):
    return " ".join(str(text).lower().replace("_", " ").replace("-", " ").split())


def _aliases(component):
    """Generated variants plus declared aliases, e.g. product-catalog / productcatalog."""
    base = component.name.lower()
    variants = {base, base.replace("-", " "), base.replace("-", ""), base.replace("-", "_")}
    variants |= {f"{v}service" for v in (base.replace("-", ""),)}
    variants |= {f"{base.replace('-', ' ')} service"}
    variants |= {str(alias).lower() for alias in component.aliases}
    return {_normalize(v) for v in variants if v}


def load_catalog(catalog_path, alerts_path=None):
    """Read and validate the YAML files; raise ValueError on any inconsistency."""
    alerts_text = Path(alerts_path).read_text(encoding="utf-8") if alerts_path else ""
    return parse_catalog(Path(catalog_path).read_text(encoding="utf-8"), alerts_text)


def parse_catalog(catalog_text, alerts_text=""):
    """Parse and validate catalog YAML text (plus optional alerts YAML text)."""
    try:
        data = yaml.safe_load(catalog_text) or {}
        alert_data = data.get("alerts", []) + (yaml.safe_load(alerts_text or "") or {}).get(
            "alerts", []
        )
    except yaml.YAMLError as error:
        raise ValueError(f"The catalog is not valid YAML: {error}") from error
    if not data.get("services"):
        raise ValueError("The catalog must list at least one service under `services`.")
    try:
        return _build_catalog(data, alert_data)
    except (KeyError, TypeError) as error:
        raise ValueError(f"The catalog is missing a required field: {error}") from error


def _build_catalog(data, alert_data):
    teams = {item["name"]: item.get("escalates_to") for item in data.get("teams", [])}
    components = {}
    entries = [("SERVICE", i) for i in data.get("services", [])]
    entries += [("DATASTORE", i) for i in data.get("datastores", [])]
    for kind, item in entries:
        if item["name"] in components:
            raise ValueError(f"Component {item['name']} is defined twice.")
        components[item["name"]] = Component(
            name=item["name"],
            kind=kind,
            team=item["team"],
            description=item.get("description", ""),
            slo=item.get("slo", ""),
            aliases=item.get("aliases", []),
            depends_on=[(d["target"], d["type"]) for d in item.get("depends_on", [])],
        )
    alerts = {
        item["name"]: {
            "service": item["service"],
            "metric": item.get("metric", ""),
            "runbook_file": item.get("runbook", {}).get("file", ""),
            "runbook_section": item.get("runbook", {}).get("section", ""),
        }
        for item in alert_data
    }
    catalog = Catalog(components=components, teams=teams, alerts=alerts)
    _validate(catalog)
    for component in components.values():
        for alias in _aliases(component):
            owner = catalog.alias_index.setdefault(alias, component.name)
            if owner != component.name:
                raise ValueError(f"Alias {alias!r} is used by {owner} and {component.name}.")
    return catalog


def _validate(catalog):
    names = list(catalog.components) + list(catalog.teams) + list(catalog.alerts)
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"Names must be unique across services, teams, alerts: {duplicates}")
    for team, target in catalog.teams.items():
        if target and target not in catalog.teams:
            raise ValueError(f"Team {team} escalates to unknown team {target}.")
    for component in catalog.components.values():
        if component.team not in catalog.teams:
            raise ValueError(f"{component.name} is owned by unknown team {component.team}.")
        for target, dep_type in component.depends_on:
            if target not in catalog.components or target == component.name:
                raise ValueError(f"{component.name} depends on invalid target {target}.")
            if dep_type not in DEPENDENCY_TYPES:
                raise ValueError(f"{component.name} -> {target}: unknown type {dep_type}.")
    for alert, info in catalog.alerts.items():
        if info["service"] not in catalog.components:
            raise ValueError(f"Alert {alert} fires on unknown service {info['service']}.")


def runbook_section_id(file_name, section):
    return f"runbook:{file_name}#{section}"


def catalog_graph(catalog):
    """Entity nodes and relations for the catalog, all tagged source=catalog."""
    tag = {"source": CATALOG_SOURCE, "file_name": CATALOG_FILE_NAME}
    nodes = [EntityNode(name=team, label="TEAM", properties=dict(tag)) for team in catalog.teams]
    relations = [
        Relation(label="ESCALATES_TO", source_id=team, target_id=target, properties=dict(tag))
        for team, target in catalog.teams.items()
        if target
    ]
    for c in catalog.components.values():
        props = {**tag, "description": c.description, "slo": c.slo, "section": c.name}
        nodes.append(EntityNode(name=c.name, label=c.kind, properties=props))
        relations.append(
            Relation(label="OWNED_BY", source_id=c.name, target_id=c.team, properties=dict(tag))
        )
        relations += [
            Relation(
                label="DEPENDS_ON",
                source_id=c.name,
                target_id=target,
                properties={**tag, "type": dep_type},
            )
            for target, dep_type in c.depends_on
        ]
    for alert, info in catalog.alerts.items():
        nodes.append(
            EntityNode(name=alert, label="ALERT", properties={**tag, "metric": info["metric"]})
        )
        relations.append(
            Relation(label="FIRES_ON", source_id=alert, target_id=info["service"], properties=tag)
        )
        if info["runbook_file"]:
            section = runbook_section_id(info["runbook_file"], info["runbook_section"])
            section_props = {
                **tag,
                "file_name": info["runbook_file"],
                "section": info["runbook_section"],
            }
            nodes.append(
                EntityNode(name=section, label="RUNBOOK_SECTION", properties=section_props)
            )
            relations.append(
                Relation(label="HANDLED_BY", source_id=alert, target_id=section, properties=tag)
            )
    return nodes, relations


CATALOG_SOURCE_ID = "catalog-source"


def save_catalog_source(graph_store, catalog_text, alerts_text=""):
    """Keep the YAML in Neo4j (not an __Entity__ node) so queries can rebuild the alias table."""
    graph_store.structured_query(
        "MERGE (c:CatalogSource {id: $id}) SET c.catalog = $catalog, c.alerts = $alerts",
        param_map={"id": CATALOG_SOURCE_ID, "catalog": catalog_text, "alerts": alerts_text},
    )


def load_catalog_from_graph(graph_store):
    """Return the imported catalog, or None if no catalog has been imported yet."""
    rows = graph_store.structured_query(
        "MATCH (c:CatalogSource {id: $id}) RETURN c.catalog AS catalog, c.alerts AS alerts",
        param_map={"id": CATALOG_SOURCE_ID},
    )
    if not rows or not rows[0]["catalog"]:
        return None
    return parse_catalog(rows[0]["catalog"], rows[0]["alerts"] or "")


def import_catalog(graph_store, catalog):
    """Replace all catalog facts in place; document-derived facts are left untouched."""
    nodes, relations = catalog_graph(catalog)
    keep = [node.id for node in nodes]
    # Catalog edges are rebuilt from scratch so removed dependencies disappear.
    graph_store.structured_query(
        "MATCH ()-[r]->() WHERE r.source = $source DELETE r", param_map={"source": CATALOG_SOURCE}
    )
    # Remove catalog nodes that left the catalog and are not referenced by any document.
    graph_store.structured_query(
        "MATCH (n:__Entity__) WHERE n.source = $source AND NOT n.id IN $keep "
        "AND size(coalesce(n.doc_ids, [])) = 0 DETACH DELETE n",
        param_map={"source": CATALOG_SOURCE, "keep": keep},
    )
    graph_store.upsert_nodes(nodes)
    graph_store.upsert_relations(relations)
    return len(nodes), len(relations)


def catalog_documents(catalog, uploaded_at=""):
    """One text document per component so the hybrid baseline sees the same facts."""
    documents = []
    for c in catalog.components.values():
        downstream = ", ".join(f"{t} ({k})" for t, k in c.depends_on) or "none"
        upstream = ", ".join(f"{t} ({k})" for t, k in catalog.upstream_of(c.name)) or "none"
        alerts = [a for a, info in catalog.alerts.items() if info["service"] == c.name]
        escalation = " -> ".join(catalog.escalation_path(c.team))
        lines = [
            f"{c.kind.title()} {c.name}: {c.description}".strip(),
            f"Aliases: {', '.join(c.aliases) or 'none'}.",
            f"Owner team: {c.team}. Escalation path: {escalation}.",
            f"Depends on: {downstream}.",
            f"Used by: {upstream}.",
            f"SLO: {c.slo or 'not specified'}.",
            f"Alerts: {', '.join(alerts) or 'none'}.",
        ]
        documents.append(
            Document(
                id_=f"catalog-{c.name}",
                text="\n".join(lines),
                metadata={
                    "file_name": CATALOG_FILE_NAME,
                    "section": c.name,
                    "doc_type": "service_catalog",
                    "source_doc_id": f"catalog-{c.name}",
                    "content_type": "text",
                    "uploaded_at": uploaded_at,
                },
                excluded_embed_metadata_keys=["source_doc_id", "uploaded_at"],
                excluded_llm_metadata_keys=["source_doc_id", "uploaded_at"],
            )
        )
    return documents
