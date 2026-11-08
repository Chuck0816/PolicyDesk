"""Command-line operations for policy indexing, retrieval, and persistent conversations."""

import argparse
import logging

from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.schema import MetadataMode

import services as functions
import storage
from base_llm import create_models
from ingestion import load_documents


def preview(functions):
    documents = load_documents()
    nodes = functions.split_documents(documents)
    print(f"Documents: {len(documents)} | Nodes: {len(nodes)}")
    for node in nodes:
        print(f"\nSource: {node.metadata['file_name']} | Source Document: {node.ref_doc_id}")
        print(node.get_content(MetadataMode.NONE))


def build_index(context, collection, functions):
    # Reject repeated builds to avoid mixing or duplicating indexed policy data.
    if context.index_store.get_index_struct(storage.INDEX_ID) or collection.count():
        raise ValueError(
            "This knowledge base already has data. Use ask/chat, or set a new KB_NAME."
        )
    nodes = functions.split_documents(load_documents())
    index = VectorStoreIndex(nodes, storage_context=context, store_nodes_override=True)
    index.set_index_id(storage.INDEX_ID)
    print(f"Saved {len(nodes)} Nodes. The index is ready; continue with ask.")
    return index


def show_sources(response):
    print("\nRetrieved sources:")
    for item in response.source_nodes:
        score = f"{item.score:.4f}" if item.score is not None else "n/a"
        print(f"- {item.metadata.get('file_name', 'unknown')} | similarity={score}")


def run(args):
    if args.command == "preview":
        preview(functions)
        return
    if args.command == "ask" and (not args.question or not args.question.strip()):
        raise ValueError('Provide a question: python main.py ask "Who approves leave?"')
    context, collection = storage.connect_storage(functions, create=args.command == "build")
    if args.command == "build" and (
        context.index_store.get_index_struct(storage.INDEX_ID) or collection.count()
    ):
        raise ValueError("Already built. Continue with ask/chat, or set a new KB_NAME.")
    llm, embed_model = create_models()
    functions.configure_settings(llm, embed_model)
    if args.command == "build":
        build_index(context, collection, functions)
        return
    index = load_index_from_storage(context, index_id=storage.INDEX_ID)
    query_engine = functions.build_query_engine(index, llm)
    if args.command == "ask":
        response = query_engine.query(args.question)
        response.print_response_stream()
        show_sources(response)
        return
    memory = functions.build_chat_memory(
        storage.connect_chat_store(), f"{storage.KB_NAME}:chat:{args.session}"
    )
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
        description="PolicyDesk: internal policy search and conversations."
    )
    parser.add_argument("command", choices=["preview", "build", "ask", "chat"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--session", default="review", help="Persistent chat session name.")
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
