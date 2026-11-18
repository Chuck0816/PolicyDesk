"""Knowledge-base operations shared by the CLI and the web demo.

Adds PDFs and Markdown, imports the service catalog, lists and deletes documents across
MongoDB, Redis, Chroma and (for the on-call domain) Neo4j, and builds the query engine for
the configured domain and retrieval mode.
"""

from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import load_index_from_storage

import catalog as service_catalog
import graph_ingestion
import graph_retrieval
import pdf_ingestion
import services
import storage

MARKDOWN_SUFFIXES = (".md", ".markdown")
CATALOG_ROW_ID = "catalog"
LISTED_TYPES = {
    pdf_ingestion.UPLOAD_DOC_TYPE: "pdf",
    "runbook": "runbook",
    "postmortem": "postmortem",
}


def is_oncall():
    return storage.KB_DOMAIN == "oncall"


def markdown_doc_type(path, text, explicit=None):
    """Use --doc-type, then front matter `doc_type`, then a runbooks/postmortems folder."""
    if explicit:
        return explicit
    front, _ = graph_ingestion.parse_front_matter(text)
    if front.get("doc_type") in graph_ingestion.DOC_TYPES:
        return front["doc_type"]
    for part in reversed(Path(path).parts[:-1]):
        for doc_type in graph_ingestion.DOC_TYPES:
            if part.lower().startswith(doc_type):
                return doc_type
    raise ValueError(
        f"Cannot tell whether {Path(path).name} is a runbook or a postmortem. "
        "Add `doc_type: runbook` or `doc_type: postmortem` to its front matter."
    )


def build_extraction(llm):
    """The schema extractor plus a Redis cache so re-adding a file does not pay again."""
    from llama_index.storage.kvstore.redis import RedisKVStore

    cache = RedisKVStore.from_host_and_port(host="127.0.0.1", port=6389)
    return graph_ingestion.build_extractor(llm), cache


def add_document(
    path,
    context,
    collection,
    *,
    work_dir,
    doc_type=None,
    graph_store=None,
    catalog=None,
    extraction=None,
    parse=None,
    progress=None,
):
    """Add a PDF or Markdown file. Returns (doc_id, chunk count, report dict)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        kwargs = {"parse": parse} if parse else {}
        doc_id, count, skipped = pdf_ingestion.ingest_pdf(
            path, context, collection, work_dir=work_dir, progress=progress, **kwargs
        )
        return doc_id, count, {"skipped": skipped}
    if suffix not in MARKDOWN_SUFFIXES:
        raise ValueError(f"Only PDF and Markdown files are supported: {path.name}")
    if not is_oncall():
        raise ValueError("Markdown runbooks and postmortems need KB_DOMAIN=oncall.")
    text = path.read_text(encoding="utf-8")
    kind = markdown_doc_type(path, text, doc_type)
    extractor, cache = extraction if extraction else (None, None)
    if kind == "postmortem" and graph_store is not None and catalog is None:
        raise ValueError("Import the service catalog before adding postmortems.")
    return graph_ingestion.ingest_markdown(
        path,
        context,
        collection,
        doc_type=kind,
        catalog=catalog,
        graph_store=graph_store,
        extractor=extractor,
        cache=cache,
        progress=progress,
    )


def import_catalog(catalog_text, alerts_text, context, collection, graph_store):
    """Replace the catalog in Neo4j and its rendered documents in the vector stores."""
    if graph_store is None:
        raise ValueError("Importing the service catalog needs Neo4j (KB_DOMAIN=oncall).")
    catalog = service_catalog.parse_catalog(catalog_text, alerts_text)
    uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    nodes = services.split_documents(service_catalog.catalog_documents(catalog, uploaded_at))
    with pdf_ingestion.WRITE_LOCK:
        service_catalog.import_catalog(graph_store, catalog)
        service_catalog.save_catalog_source(graph_store, catalog_text, alerts_text)
        old = [
            doc_id
            for doc_id in (context.docstore.get_all_ref_doc_info() or {})
            if doc_id.startswith("catalog-")
        ]
        if old:
            index = load_index_from_storage(
                context, index_id=storage.INDEX_ID, store_nodes_override=True
            )
            for doc_id in old:
                index.delete_ref_doc(doc_id, delete_from_docstore=True)
        graph_ingestion.append_to_vector_index(context, collection, nodes)
    return catalog, len(nodes)


def list_documents(docstore):
    """One row per document, newest first; catalog documents are grouped into one row."""
    rows, catalog_row = [], None
    for doc_id, info in (docstore.get_all_ref_doc_info() or {}).items():
        metadata = info.metadata or {}
        if metadata.get("doc_type") == "service_catalog":
            if catalog_row is None:
                catalog_row = {
                    "doc_id": CATALOG_ROW_ID,
                    "file_name": service_catalog.CATALOG_FILE_NAME,
                    "doc_type": "service_catalog",
                    "uploaded_at": "",
                    "chunks": 0,
                    "services": 0,
                    "skipped_figures": 0,
                    "graph_relations": 0,
                }
            catalog_row["chunks"] += len(info.node_ids)
            catalog_row["services"] += 1
            catalog_row["uploaded_at"] = max(
                catalog_row["uploaded_at"], metadata.get("uploaded_at", "")
            )
            continue
        kind = LISTED_TYPES.get(metadata.get("doc_type"))
        if kind is None:
            continue
        rows.append(
            {
                "doc_id": doc_id,
                "file_name": metadata.get("file_name", doc_id),
                "doc_type": kind,
                "uploaded_at": metadata.get("uploaded_at", ""),
                "chunks": len(info.node_ids),
                "skipped_figures": metadata.get("skipped_figures", 0),
                "graph_relations": metadata.get("graph_relations", 0),
            }
        )
    rows.sort(key=lambda row: row["uploaded_at"], reverse=True)
    return ([catalog_row] if catalog_row else []) + rows


def delete_document(context, doc_id, graph_store=None):
    """Remove a document from every store; graph facts other documents support are kept."""
    if doc_id == CATALOG_ROW_ID:
        raise ValueError("The service catalog cannot be deleted; import a new version instead.")
    if not context.docstore.ref_doc_exists(doc_id):
        raise ValueError(f"{doc_id} is not in the knowledge base.")
    if graph_store is not None:
        with pdf_ingestion.WRITE_LOCK:
            graph_ingestion.delete_document_graph(graph_store, doc_id)
    pdf_ingestion.delete_pdf(context, doc_id)


def connect_graph():
    """(graph_store, catalog) for the on-call domain; (None, None) for policy.

    Raises if NEO4J_PASSWORD is missing or Neo4j is unreachable; callers that can fall back
    to hybrid retrieval catch the error.
    """
    if not is_oncall():
        return None, None
    graph_store = storage.connect_graph_store()
    return graph_store, service_catalog.load_catalog_from_graph(graph_store)


def graph_search_for(graph_store, catalog):
    """GraphSearch when the catalog is in Neo4j; None means graph modes fall back to hybrid."""
    if graph_store is None or catalog is None:
        return None
    return graph_retrieval.GraphSearch(graph_store, catalog)


def build_query_engine(index, llm, reranker, mode, graph_search=None):
    """Policy domain: the existing engine. On-call: the requested mode (hybrid if no graph)."""
    if not is_oncall():
        return services.build_query_engine(index, llm, advanced=True, reranker=reranker)
    resolved = graph_retrieval.resolve_mode(mode, graph_search is not None)
    return graph_retrieval.build_oncall_query_engine(index, llm, reranker, resolved, graph_search)
