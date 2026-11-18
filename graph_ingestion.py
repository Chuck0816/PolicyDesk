"""Markdown ingestion for runbooks and postmortems, with incident-graph extraction.

Every chunk is split once and written with the same node ID to MongoDB, Redis, Chroma and
(for postmortems) Neo4j, so a graph edge can always be traced back to its source text.
Provenance lives on the graph itself: each extracted edge and entity keeps `doc_ids` and
`chunk_ids` lists, because Neo4j merges an edge mentioned by several documents into one.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml
from llama_index.core import Document, VectorStoreIndex, load_index_from_storage
from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    EntityNode,
    Relation,
)
from llama_index.core.indices.property_graph import PropertyGraphIndex, SchemaLLMPathExtractor
from llama_index.core.schema import (
    NodeRelationship,
    ObjectType,
    RelatedNodeInfo,
    TransformComponent,
)

import services
import storage
from graph_schema import (
    CATALOG_SOURCE,
    EXTRACTION_SCHEMA,
    RELATION_PROPS,
    ExtractedEntity,
    ExtractedRelation,
    canonical_symptom,
)
from pdf_ingestion import WRITE_LOCK

DOC_TYPES = ("runbook", "postmortem")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
EXTRACTION_VERSION = "oncall-v1"  # Bump when the schema or prompt changes to bust the cache.
# Upload bookkeeping kept out of embeddings and prompts.
BOOKKEEPING_KEYS = ["source_doc_id", "uploaded_at", "graph_relations"]


def parse_front_matter(text):
    """Split optional `---` YAML front matter (service, incident_id, title) from the body."""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return yaml.safe_load(text[4:end]) or {}, text[end + 4 :].lstrip("\n")
    return {}, text


def split_sections(body):
    """Return (section path, text) pairs split on Markdown headings outside code fences."""
    sections, path, lines, fenced = [], [], [], False

    def flush():
        content = "\n".join(lines).strip()
        if content:
            sections.append((" > ".join(title for _, title in path) or "Introduction", content))
        lines.clear()

    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        match = None if fenced else HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path = [(lvl, title) for lvl, title in path if lvl < level]
            path.append((level, match.group(2)))
        lines.append(line)
    flush()
    return sections


def markdown_nodes(text, *, file_name, doc_type, doc_id):
    """Section-aware chunks with the metadata both the vector and graph indexes need."""
    if doc_type not in DOC_TYPES:
        raise ValueError(f"doc_type must be one of {DOC_TYPES}, not {doc_type!r}.")
    front, body = parse_front_matter(text)
    if doc_type == "postmortem" and not front.get("incident_id"):
        raise ValueError(f"{file_name}: postmortems need `incident_id` in their front matter.")
    shared = {
        "file_name": file_name,
        "doc_type": doc_type,
        "source_doc_id": doc_id,
        "content_type": "text",
        "title": str(front.get("title", Path(file_name).stem)),
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for key in ("service", "incident_id"):
        if front.get(key):
            shared[key] = str(front[key])
    documents = [
        Document(
            id_=f"{doc_id}#s{number}",
            text=content,
            metadata={**shared, "section": section},
            excluded_embed_metadata_keys=[*BOOKKEEPING_KEYS, "incident_id"],
            excluded_llm_metadata_keys=BOOKKEEPING_KEYS,
        )
        for number, (section, content) in enumerate(split_sections(body))
    ]
    if not documents:
        raise ValueError(f"{file_name} has no text.")
    nodes = services.split_documents(documents)
    source = RelatedNodeInfo(node_id=doc_id, node_type=ObjectType.DOCUMENT)
    for node in nodes:
        node.relationships[NodeRelationship.SOURCE] = source
    return nodes


def build_extractor(llm):
    """Schema-constrained extractor; strict=True drops anything outside the schema."""
    return SchemaLLMPathExtractor(
        llm=llm,
        possible_entities=ExtractedEntity,
        possible_relations=ExtractedRelation,
        kg_validation_schema=EXTRACTION_SCHEMA,
        possible_relation_props=RELATION_PROPS,
        strict=True,
        max_triplets_per_chunk=15,
        num_workers=4,
    )


def align_chunk_graph(catalog, metadata, kg_nodes, kg_relations):
    """Map extracted names onto catalog and vocabulary names; drop what cannot be aligned.

    Returns (nodes, relations, dropped) where dropped lists "LABEL:name" strings.
    """
    aligned, dropped = {}, []
    for node in kg_nodes:
        name = None
        if node.label in ("SERVICE", "DATASTORE"):
            name = catalog.resolve(node.name)
            label = catalog.components[name].kind if name else node.label
        elif node.label == "INCIDENT":
            # One postmortem describes one incident; its ID comes from front matter.
            name, label = metadata.get("incident_id"), "INCIDENT"
        elif node.label == "SYMPTOM":
            name, label = canonical_symptom(node.name), "SYMPTOM"
        elif node.label == "MITIGATION":
            name, label = " ".join(node.name.split())[:160] or None, "MITIGATION"
        if name is None:
            dropped.append(f"{node.label}:{node.name}")
            continue
        aligned[node.id] = EntityNode(name=name, label=label, properties=dict(node.properties))

    relations = []
    for rel in kg_relations:
        source, target = aligned.get(rel.source_id), aligned.get(rel.target_id)
        if source is None or target is None:
            continue
        properties = dict(rel.properties)
        if properties.get("service"):
            resolved = catalog.resolve(properties["service"])
            if resolved:
                properties["service"] = resolved
            else:
                properties.pop("service")
        relations.append(
            Relation(
                label=rel.label, source_id=source.id, target_id=target.id, properties=properties
            )
        )
    nodes = list({node.id: node for node in aligned.values()}.values())
    return nodes, relations, dropped


def _serialize(nodes, relations):
    return json.dumps(
        {
            "nodes": [[n.label, n.name, n.properties] for n in nodes],
            "relations": [[r.label, r.source_id, r.target_id, r.properties] for r in relations],
        },
        ensure_ascii=False,
        default=str,
    )


def _deserialize(raw):
    data = json.loads(raw)
    nodes = [
        EntityNode(label=lbl, name=name, properties=props) for lbl, name, props in data["nodes"]
    ]
    relations = [
        Relation(label=lbl, source_id=s, target_id=t, properties=props)
        for lbl, s, t, props in data["relations"]
    ]
    return nodes, relations


def extract_incident_graph(nodes, extractor, catalog, cache=None, cache_namespace="kg_cache"):
    """Run (or reuse cached) LLM extraction, then align. Returns per-chunk graphs and a report.

    The cache is any llama-index KV store; keys depend on the chunk text, the extraction
    version, and the model, so changing the schema or model re-extracts.
    """
    model = getattr(extractor.llm, "model_name", type(extractor.llm).__name__)

    def key(node):
        raw = f"{EXTRACTION_VERSION}|{model}|{node.get_content()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    raw_graphs, misses = {}, []
    for node in nodes:
        hit = cache.get(key(node), collection=cache_namespace) if cache else None
        if hit:
            raw_graphs[node.id_] = _deserialize(hit["graph"])
        else:
            misses.append(node)
    if misses:
        # The extractor only adds metadata, so the original nodes are not rebuilt.
        for node in extractor(misses):
            graph = (node.metadata.pop(KG_NODES_KEY, []), node.metadata.pop(KG_RELATIONS_KEY, []))
            raw_graphs[node.id_] = graph
            if cache:
                cache.put(key(node), {"graph": _serialize(*graph)}, collection=cache_namespace)

    graphs, report = {}, {"chunks": len(nodes), "empty_chunks": 0, "dropped": []}
    for node in nodes:
        kg_nodes, kg_relations = raw_graphs[node.id_]
        aligned_nodes, aligned_relations, dropped = align_chunk_graph(
            catalog, node.metadata, kg_nodes, kg_relations
        )
        report["dropped"] += dropped
        report["empty_chunks"] += not aligned_relations
        graphs[node.id_] = (aligned_nodes, aligned_relations)
    return graphs, report


class AlreadyExtracted(TransformComponent):
    """No-op extractor. An empty kg_extractors list is falsy, and PropertyGraphIndex would
    then run its defaults: an unconstrained LLM extractor plus chunk-to-document links."""

    def __call__(self, nodes, **kwargs):
        return nodes


def write_incident_graph(graph_store, nodes, graphs, doc_id):
    """Insert chunks and aligned facts through PropertyGraphIndex, then record provenance."""
    for node in nodes:
        kg_nodes, kg_relations = graphs[node.id_]
        node.metadata[KG_NODES_KEY] = kg_nodes
        node.metadata[KG_RELATIONS_KEY] = kg_relations
    # Extraction already ran; the index only writes chunks, entities and relations.
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        kg_extractors=[AlreadyExtracted()],
        embed_kg_nodes=False,
        use_async=False,
    )
    index.insert_nodes(nodes)

    relation_rows, entity_rows = [], []
    for node in nodes:
        kg_nodes, kg_relations = graphs[node.id_]
        entity_rows += [{"id": n.id, "doc_id": doc_id} for n in kg_nodes]
        relation_rows += [
            {
                "source": r.source_id,
                "label": r.label,
                "target": r.target_id,
                "doc_id": doc_id,
                "chunk_id": node.id_,
            }
            for r in kg_relations
        ]
    graph_store.structured_query(
        """
        UNWIND $rows AS row
        MATCH (s:__Node__ {id: row.source})-[r]->(t:__Node__ {id: row.target})
        WHERE type(r) = row.label
        SET r.doc_ids = apoc.coll.toSet(coalesce(r.doc_ids, []) + row.doc_id),
            r.chunk_ids = apoc.coll.toSet(coalesce(r.chunk_ids, []) + row.chunk_id)
        """,
        param_map={"rows": relation_rows},
    )
    graph_store.structured_query(
        """
        UNWIND $rows AS row
        MATCH (e:__Entity__ {id: row.id})
        SET e.doc_ids = apoc.coll.toSet(coalesce(e.doc_ids, []) + row.doc_id)
        """,
        param_map={"rows": entity_rows},
    )
    return len(relation_rows)


def delete_document_graph(graph_store, doc_id):
    """Remove one document's chunks and only the facts no other document still supports."""
    chunk_rows = graph_store.structured_query(
        "MATCH (c:Chunk) WHERE c.ref_doc_id = $doc RETURN c.id AS id", param_map={"doc": doc_id}
    )
    chunk_ids = [row["id"] for row in chunk_rows]
    graph_store.structured_query(
        """
        MATCH ()-[r]->() WHERE $doc IN coalesce(r.doc_ids, [])
        SET r.doc_ids = [x IN r.doc_ids WHERE x <> $doc],
            r.chunk_ids = [x IN coalesce(r.chunk_ids, []) WHERE NOT x IN $chunks]
        WITH r WHERE size(r.doc_ids) = 0
        DELETE r
        """,
        param_map={"doc": doc_id, "chunks": chunk_ids},
    )
    graph_store.structured_query(
        """
        MATCH (e:__Entity__) WHERE $doc IN coalesce(e.doc_ids, [])
        SET e.doc_ids = [x IN e.doc_ids WHERE x <> $doc]
        """,
        param_map={"doc": doc_id},
    )
    graph_store.structured_query(
        "MATCH (c:Chunk) WHERE c.id IN $chunks DETACH DELETE c", param_map={"chunks": chunk_ids}
    )
    # Extracted entities nobody mentions any more; catalog entities are never removed here.
    graph_store.structured_query(
        """
        MATCH (e:__Entity__)
        WHERE coalesce(e.source, '') <> $catalog AND size(coalesce(e.doc_ids, [])) = 0
        DETACH DELETE e
        """,
        param_map={"catalog": CATALOG_SOURCE},
    )
    return len(chunk_ids)


def graph_has_catalog(graph_store):
    rows = graph_store.structured_query("MATCH (s:SERVICE) RETURN count(s) > 0 AS ready")
    return bool(rows and rows[0]["ready"])


def append_to_vector_index(context, collection, nodes):
    """Same incremental write as PDF ingestion: MongoDB, Redis and Chroma stay in sync."""
    if context.index_store.get_index_struct(storage.INDEX_ID):
        index = load_index_from_storage(
            context, index_id=storage.INDEX_ID, store_nodes_override=True
        )
        index.insert_nodes(nodes)
        return index
    if collection.count():
        raise ValueError("Chroma has data but no index structure; use a new KB_NAME.")
    index = VectorStoreIndex(nodes, storage_context=context, store_nodes_override=True)
    index.set_index_id(storage.INDEX_ID)
    return index


def ingest_markdown(
    path,
    context,
    collection,
    *,
    doc_type,
    catalog=None,
    graph_store=None,
    extractor=None,
    cache=None,
    progress=None,
):
    """Add one runbook or postmortem. Postmortems also add incident facts to Neo4j.

    Returns (doc_id, chunk count, report). LLM extraction runs before anything is written,
    so a failed extraction leaves every store unchanged.
    """

    def report_step(step):
        if progress is not None:
            progress(step)

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    doc_id = "md-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    if context.docstore.ref_doc_exists(doc_id):
        raise ValueError(f"{path.name} is already in the knowledge base ({doc_id}).")

    report_step("parse")
    nodes = markdown_nodes(text, file_name=path.name, doc_type=doc_type, doc_id=doc_id)
    graphs, report = None, {"chunks": len(nodes)}
    if doc_type == "postmortem" and graph_store is not None:
        if catalog is None or extractor is None:
            raise ValueError("Postmortem graph extraction needs the catalog and an extractor.")
        report_step("extract")
        graphs, report = extract_incident_graph(nodes, extractor, catalog, cache)
        relation_count = sum(len(relations) for _, relations in graphs.values())
        for node in nodes:
            node.metadata["graph_relations"] = relation_count  # Shown in the document list.

    report_step("index")
    with WRITE_LOCK:
        if context.docstore.ref_doc_exists(doc_id):
            raise ValueError(f"{path.name} is already in the knowledge base ({doc_id}).")
        index = append_to_vector_index(context, collection, nodes)
        if graphs is not None:
            try:
                report["relations"] = write_incident_graph(graph_store, nodes, graphs, doc_id)
            except Exception:
                # Undo partial graph writes and the vector insert so no store is half-updated.
                delete_document_graph(graph_store, doc_id)
                index.delete_ref_doc(doc_id, delete_from_docstore=True)
                raise
    return doc_id, len(nodes), report
