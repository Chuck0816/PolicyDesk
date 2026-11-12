"""Command-line operations for uploaded PDFs, questions, and persistent conversations."""

import argparse
import logging
import tempfile
from pathlib import Path

from llama_index.core import load_index_from_storage
from llama_index.core.chat_engine import CondenseQuestionChatEngine

import pdf_ingestion
import services as functions
import storage
from base_llm import create_models, create_reranker

USAGE = {
    "add": "Provide a PDF: python main.py add /path/to/policy.pdf",
    "delete": "Provide a document ID from `python main.py list`: python main.py delete pdf-...",
    "ask": 'Provide a question: python main.py ask "Who approves leave?"',
}
EMPTY_KNOWLEDGE_BASE = "The knowledge base has no PDFs. Add one: python main.py add <file.pdf>"


def show_documents(documents):
    if not documents:
        print(EMPTY_KNOWLEDGE_BASE)
        return
    for document in documents:
        skipped = document["skipped_figures"]
        note = f" | {skipped} figure(s) skipped" if skipped else ""
        print(
            f"{document['doc_id']} | {document['file_name']} | {document['chunks']} chunks"
            f" | added {document['uploaded_at']}{note}"
        )


def show_sources(response):
    print("\nRetrieved sources:")
    for item in response.source_nodes:
        score = f"{item.score:.4f}" if item.score is not None else "n/a"
        name = item.metadata.get("file_name", "unknown")
        page = item.metadata.get("page", "?")
        kind = item.metadata.get("content_type", "text")
        print(f"- {name} | page {page} | {kind} | reranker_score={score}")


def add_pdf(args, context, collection):
    def existing_content_list(pdf, output_dir):
        return Path(args.content_list)

    parse = existing_content_list if args.content_list else pdf_ingestion.run_mineru
    with tempfile.TemporaryDirectory() as work_dir:
        doc_id, count, skipped = pdf_ingestion.ingest_pdf(
            args.target, context, collection, work_dir=work_dir, parse=parse
        )
    print(f"Added {doc_id}: {count} chunks in {storage.KB_NAME}.")
    for item in skipped:
        caption = f" ({item['caption']})" if item["caption"] else ""
        print(f"Skipped figure on page {item['page']}{caption}: {item['reason']}")
    if skipped:
        print("Skipped figures are not searchable; the rest of the PDF was added.")


def run(args):
    target = (args.target or "").strip()
    if args.command in USAGE and not target:
        raise ValueError(USAGE[args.command])
    if args.content_list and args.command != "add":
        raise ValueError("--content-list applies only to add.")
    if args.command == "list":
        show_documents(pdf_ingestion.list_pdfs(storage.connect_docstore()))
        return
    if args.command in {"ask", "chat"} and not pdf_ingestion.list_pdfs(storage.connect_docstore()):
        raise ValueError(EMPTY_KNOWLEDGE_BASE)

    # Only adding may create the Chroma collection of a new, empty knowledge base.
    context, collection = storage.connect_storage(functions, create=args.command == "add")
    llm, embed_model = create_models()
    functions.configure_settings(llm, embed_model)
    if args.command == "add":
        add_pdf(args, context, collection)
        return
    if args.command == "delete":
        pdf_ingestion.delete_pdf(context, target)
        print(f"Deleted {target} from {storage.KB_NAME}.")
        return

    index = load_index_from_storage(context, index_id=storage.INDEX_ID)
    reranker = create_reranker()
    query_engine = functions.build_query_engine(index, llm, advanced=True, reranker=reranker)
    if args.command == "ask":
        response = query_engine.query(args.target)
        response.print_response_stream()
        show_sources(response)
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
            response = chat_engine.chat(question)
            print(response)
            show_sources(response)


if __name__ == "__main__":
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(
        description="PolicyDesk: manage uploaded policy PDFs and ask questions about them."
    )
    parser.add_argument("command", choices=["add", "delete", "list", "ask", "chat"])
    parser.add_argument(
        "target", nargs="?", help="PDF path for add, document ID for delete, question for ask."
    )
    parser.add_argument("--content-list", help="add: reuse an existing MinerU *_content_list.json.")
    parser.add_argument("--session", default="review", help="Persistent chat session name.")
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
