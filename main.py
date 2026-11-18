"""Command-line operations for documents, the service catalog, questions, and conversations."""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from llama_index.core import load_index_from_storage
from llama_index.core.chat_engine import CondenseQuestionChatEngine

import graph_retrieval
import kb
import services as functions
import storage
from base_llm import create_models, create_reranker
from metrics import TOKEN_METER, AnswerTimer, format_seconds, log_answer

USAGE = {
    "add": "Provide a file: python main.py add /path/to/policy.pdf (or runbook.md)",
    "delete": "Provide a document ID from `python main.py list`: python main.py delete pdf-...",
    "ask": 'Provide a question: python main.py ask "Who approves leave?"',
    "import-catalog": "Provide the catalog: python main.py import-catalog service_catalog.yaml",
}
EMPTY_KNOWLEDGE_BASE = "The knowledge base has no documents. Add one: python main.py add <file>"


def show_documents(documents):
    if not documents:
        print(EMPTY_KNOWLEDGE_BASE)
        return
    for document in documents:
        notes = []
        if document.get("services"):
            notes.append(f"{document['services']} services")
        if document["skipped_figures"]:
            notes.append(f"{document['skipped_figures']} figure(s) skipped")
        if document.get("graph_relations"):
            notes.append(f"{document['graph_relations']} graph relations")
        note = "".join(f" | {item}" for item in notes)
        print(
            f"{document['doc_id']} | {document['file_name']} | {document['doc_type']} | "
            f"{document['chunks']} chunks | added {document['uploaded_at']}{note}"
        )


def show_sources(response):
    documents = [n for n in response.source_nodes if n.metadata.get("content_type") != "graph_path"]
    graph = [n for n in response.source_nodes if n.metadata.get("content_type") == "graph_path"]
    print("\nDocument evidence:")
    for item in documents:
        score = f"{item.score:.4f}" if item.score is not None else "n/a"
        name = item.metadata.get("file_name", "unknown")
        where = item.metadata.get("section") or f"page {item.metadata.get('page', '?')}"
        kind = item.metadata.get("content_type", "text")
        print(f"- {name} | {where} | {kind} | reranker_score={score}")
    if graph:
        print("\nGraph reasoning:")
        for item in graph:
            print(f"- {item.get_content()}")


def show_timing(timer, mode):
    timing, tokens = timer.summary(), TOKEN_METER.snapshot()
    log_answer(mode, timing, tokens)
    print(
        f"\nTiming ({mode}): retrieval {format_seconds(timing['retrieval_s'])}s | "
        f"first token {format_seconds(timing['first_token_s'])}s | "
        f"generation {format_seconds(timing['generation_s'])}s | "
        f"total {format_seconds(timing['total_s'])}s | "
        f"LLM calls {tokens['llm_calls']} | "
        f"~{tokens['prompt_tokens']}+{tokens['completion_tokens']} tokens (estimated)"
    )


def connect_graph_or_fallback():
    """Questions still work without Neo4j: graph modes fall back to hybrid."""
    try:
        return kb.connect_graph()
    except Exception as error:
        print(f"Graph retrieval unavailable ({error}); using hybrid retrieval.", file=sys.stderr)
        return None, None


def add_file(args, context, collection, llm, graph_store, catalog):
    def existing_content_list(pdf, output_dir):
        return Path(args.content_list)

    extraction = kb.build_extraction(llm) if graph_store is not None else None
    with tempfile.TemporaryDirectory() as work_dir:
        doc_id, count, report = kb.add_document(
            args.target,
            context,
            collection,
            work_dir=work_dir,
            doc_type=args.doc_type,
            graph_store=graph_store,
            catalog=catalog,
            extraction=extraction,
            parse=existing_content_list if args.content_list else None,
        )
    print(f"Added {doc_id}: {count} chunks in {storage.KB_NAME}.")
    for item in report.get("skipped", []):
        caption = f" ({item['caption']})" if item["caption"] else ""
        print(f"Skipped figure on page {item['page']}{caption}: {item['reason']}")
    if report.get("skipped"):
        print("Skipped figures are not searchable; the rest of the PDF was added.")
    if "relations" in report:
        print(
            f"Graph: {report['relations']} relations; {report['empty_chunks']} chunk(s) "
            f"without facts; dropped (not in the catalog): {report['dropped'] or 'none'}."
        )


def run(args):
    target = (args.target or "").strip()
    if args.command in USAGE and not target:
        raise ValueError(USAGE[args.command])
    if args.content_list and args.command != "add":
        raise ValueError("--content-list applies only to add.")
    if args.doc_type and args.command != "add":
        raise ValueError("--doc-type applies only to add.")
    if args.alerts and args.command != "import-catalog":
        raise ValueError("--alerts applies only to import-catalog.")
    if args.command == "list":
        show_documents(kb.list_documents(storage.connect_docstore()))
        return
    if args.command in {"ask", "chat"} and not kb.list_documents(storage.connect_docstore()):
        raise ValueError(EMPTY_KNOWLEDGE_BASE)
    if args.command == "import-catalog" and not kb.is_oncall():
        raise ValueError("import-catalog needs KB_DOMAIN=oncall in .env.")

    # Only writes may create the Chroma collection of a new, empty knowledge base.
    writes = args.command in {"add", "import-catalog"}
    context, collection = storage.connect_storage(functions, create=writes)
    llm, embed_model = create_models()
    functions.configure_settings(llm, embed_model)
    TOKEN_METER.attach(llm)
    if args.command in {"add", "delete", "import-catalog"}:
        # Writes in the on-call domain must reach Neo4j too, so there is no fallback here.
        graph_store, catalog = kb.connect_graph()
    else:
        graph_store, catalog = connect_graph_or_fallback()

    if args.command == "add":
        add_file(args, context, collection, llm, graph_store, catalog)
        return
    if args.command == "delete":
        kb.delete_document(context, target, graph_store)
        print(f"Deleted {target} from {storage.KB_NAME}.")
        return
    if args.command == "import-catalog":
        alerts = Path(args.alerts).read_text(encoding="utf-8") if args.alerts else ""
        imported, chunks = kb.import_catalog(
            Path(target).read_text(encoding="utf-8"), alerts, context, collection, graph_store
        )
        print(
            f"Imported {len(imported.components)} components, {len(imported.teams)} teams, "
            f"{len(imported.alerts)} alerts; {chunks} catalog chunks in {storage.KB_NAME}."
        )
        return

    index = load_index_from_storage(context, index_id=storage.INDEX_ID)
    reranker = create_reranker()
    graph_search = kb.graph_search_for(graph_store, catalog)
    mode = graph_retrieval.resolve_mode(args.mode, graph_search is not None)
    if kb.is_oncall() and mode != args.mode:
        print(f"Using {mode}: no service catalog in the graph yet.", file=sys.stderr)
    query_engine = kb.build_query_engine(index, llm, reranker, mode, graph_search)
    if args.command == "ask":
        TOKEN_METER.reset()
        timer = AnswerTimer()
        response = query_engine.query(args.target)
        timer.mark_retrieved()
        for chunk in timer.stream(response.response_gen):
            print(chunk, end="", flush=True)
        print()
        show_sources(response)
        show_timing(timer, mode if kb.is_oncall() else "hybrid")
        return
    session_key = f"{storage.KB_NAME}:chat:{args.session}"
    memory = functions.build_chat_memory(storage.connect_chat_store(), session_key)
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine, llm=llm, memory=memory
    )
    print(f"Restored {len(chat_engine.chat_history)} messages. Type /exit to finish.")
    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            break
        if question == "/exit":
            break
        if question:
            TOKEN_METER.reset()
            timer = AnswerTimer()
            # Chat replies are printed whole so an empty answer never enters the history.
            response = chat_engine.chat(question)
            timer.mark_retrieved()
            print("".join(timer.stream([str(response)])))
            show_sources(response)
            show_timing(timer, mode if kb.is_oncall() else "hybrid")


if __name__ == "__main__":
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(
        description="PolicyDesk: manage documents and ask questions about them."
    )
    parser.add_argument(
        "command", choices=["add", "delete", "list", "ask", "chat", "import-catalog"]
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="File for add, document ID for delete, question for ask, YAML for import-catalog.",
    )
    parser.add_argument("--content-list", help="add: reuse an existing MinerU *_content_list.json.")
    parser.add_argument(
        "--doc-type",
        choices=["runbook", "postmortem"],
        help="add: Markdown type when the front matter and folder do not say.",
    )
    parser.add_argument("--alerts", help="import-catalog: optional alerts.yaml.")
    parser.add_argument(
        "--mode",
        choices=list(graph_retrieval.MODES),
        default="hybrid+graph",
        help="On-call retrieval mode; falls back to hybrid when the graph has no catalog.",
    )
    parser.add_argument("--session", default="review", help="Persistent chat session name.")
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
