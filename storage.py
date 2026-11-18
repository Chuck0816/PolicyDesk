"""Database connections for policy nodes, vector indexes, and conversation history."""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv(Path(__file__).resolve().parent / ".env")
KB_NAME = os.getenv("KB_NAME", "policydesk_uploads_v1")
# "policy" keeps the PDF policy assistant; "oncall" adds Markdown, the service catalog,
# and graph retrieval. One process serves one knowledge base and one domain.
KB_DOMAIN = os.getenv("KB_DOMAIN", "policy")
if KB_DOMAIN not in ("policy", "oncall"):
    raise ValueError(f"KB_DOMAIN must be 'policy' or 'oncall', not {KB_DOMAIN!r}.")
MONGO_DB = os.getenv("MONGO_DB", "policydesk")
INDEX_ID = "policy_index"
REDIS_URL = "redis://127.0.0.1:6389/0"
MONGO_URI = "mongodb://127.0.0.1:27028/?serverSelectionTimeoutMS=5000"


def connect_docstore():
    """Open only the node store; listing documents needs neither Redis nor Chroma."""
    return MongoDocumentStore.from_uri(uri=MONGO_URI, db_name=MONGO_DB, namespace=KB_NAME)


def connect_storage(functions, create=False):
    docstore = connect_docstore()

    index_store = RedisIndexStore.from_host_and_port(host="127.0.0.1", port=6389, namespace=KB_NAME)

    client = chromadb.HttpClient(host="127.0.0.1", port=8018)
    # Query paths must not silently create an empty knowledge base.
    collection = (
        client.get_or_create_collection(KB_NAME) if create else client.get_collection(KB_NAME)
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)

    return functions.build_storage_context(docstore, index_store, vector_store), collection


def connect_chat_store():
    return RedisChatStore(redis_url=REDIS_URL)


def connect_graph_store():
    """Neo4j for the on-call graph; imported lazily so the policy domain does not need it."""
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError("Set NEO4J_PASSWORD in .env to use graph retrieval.")
    return Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USER", "neo4j"),
        password=password,
        url=os.getenv("NEO4J_URL", "bolt://127.0.0.1:7687"),
    )
