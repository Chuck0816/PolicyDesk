"""Command-line operations for policy indexing, retrieval, and persistent conversations."""

import argparse
import logging

from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.schema import MetadataMode

import services as functions
import storage
from base_llm import create_models, create_reranker
from ingestion import POLICY_CATEGORIES, load_documents


def preview(functions):
    documents = load_documents()
    nodes = functions.split_documents(documents)
    print(f"Documents: {len(documents)} | Nodes: {len(nodes)}")
    for node in nodes:
        print(f"\nSource: {node.metadata['file_name']} | Source Document: {node.ref_doc_id}")
        print(f"Metadata: {node.metadata}")
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
        name = item.metadata.get("file_name", "unknown")
        policy_id = item.metadata.get("policy_id", "unknown")
        category = item.metadata.get("category", "unknown")
        print(f"- {name} | {policy_id} | {category} | reranker_score={score}")


def run(args):
    category = args.category
    if category is not None and args.command in {"preview", "build"}:
        raise ValueError("--category applies only to ask and chat; build indexes all policies.")
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
    reranker = create_reranker()

    query_engine = functions.build_query_engine(
        index,
        llm,
        advanced=True,
        reranker=reranker,
        category=category,
    )
    if args.command == "ask":
        response = query_engine.query(args.question)
        response.print_response_stream()
        show_sources(response)
        return
    session_key = f"{storage.KB_NAME}:chat:{args.session}"
    if category is not None:
        # Scoped histories must not inject earlier answers from another policy category.
        session_key += f":category:{category}"
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
        description="PolicyDesk: internal policy search and conversations."
    )
    parser.add_argument("command", choices=["preview", "build", "ask", "chat"])
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "--category",
        choices=sorted(set(POLICY_CATEGORIES.values())),
        help="Limit ask/chat to one policy category; omit to search all policies.",
    )
    parser.add_argument("--session", default="review", help="Persistent chat session name.")
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
