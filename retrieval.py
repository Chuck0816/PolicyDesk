"""Scoped query preparation and hybrid retrieval over saved policy nodes."""

import logging
import re
from collections import Counter

from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import FilterCondition, MetadataFilter, MetadataFilters

from ingestion import POLICY_CATEGORIES
from prompts import QUERY_REWRITE_PROMPT

LOGGER = logging.getLogger(__name__)
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")
# Each Chinese character is one keyword; other letters and digits keep whole-word tokens.
BM25_TOKEN_PATTERN = r"(?u)[一-鿿]|[^\W一-鿿]+"
CANDIDATE_TOP_K = 8
FUSION_TOP_K = 10


def build_metadata_filters(category: str | None = None):
    """Return the AND scope shared by semantic and keyword retrieval."""
    if category is None:
        return None
    if category not in POLICY_CATEGORIES.values():
        raise ValueError(f"Unknown policy category: {category!r}.")
    return MetadataFilters(
        filters=[
            MetadataFilter(key="category", value=category),
            MetadataFilter(key="doc_type", value="company_policy"),
        ],
        condition=FilterCondition.AND,
    )


def select_policy_nodes(nodes, category: str | None = None):
    """Select BM25's corpus before ranking; exclude general guidance from scopes."""
    build_metadata_filters(category)
    if category is None:
        return nodes
    return [
        node
        for node in nodes
        if node.metadata.get("category") == category
        and node.metadata.get("doc_type") == "company_policy"
    ]


class QueryRewriteRetriever(BaseRetriever):
    """Rewrite retrieval text while preserving the question used for synthesis."""

    def __init__(self, base_retriever: BaseRetriever, llm):
        super().__init__(callback_manager=base_retriever.callback_manager)
        self._base_retriever = base_retriever
        self._llm = llm

    def _predict_rewrite(self, original: str) -> str:
        return self._llm.predict(QUERY_REWRITE_PROMPT, query=original).strip()

    def _rewrite(self, original: str) -> str:
        try:
            rewritten = self._predict_rewrite(original).strip()
        except NotImplementedError:
            # Unsupported provider implementations must fail explicitly.
            raise
        except Exception:
            LOGGER.warning("Query rewrite failed; using the original question.")
            return original

        # Preserve every numeric occurrence in amounts, thresholds, dates, and grades.
        if (
            not rewritten
            or "\n" in rewritten
            or len(rewritten) > 2000
            or Counter(NUMBER_PATTERN.findall(original))
            != Counter(NUMBER_PATTERN.findall(rewritten))
        ):
            return original
        return rewritten

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        original = query_bundle.query_str.strip()
        if not original:
            raise ValueError("The question must not be empty.")
        rewritten = self._rewrite(original)
        if rewritten == original:
            return self._base_retriever.retrieve(query_bundle)

        nodes = self._base_retriever.retrieve(rewritten)
        if nodes:
            return nodes
        # Retrying the original text must preserve the already selected scope.
        return self._base_retriever.retrieve(query_bundle)


def load_index_nodes(index):
    """Reuse saved node identities so fusion can deduplicate both search branches."""
    node_ids = list(dict.fromkeys(index.index_struct.nodes_dict.values()))
    if not node_ids:
        raise ValueError(
            "The index has no saved node references. Build a new knowledge "
            "base with store_nodes_override=True."
        )
    return index.docstore.get_nodes(node_ids)


def build_bm25_retriever(nodes, top_k: int):
    """Build an in-memory keyword index from database-backed policy nodes."""
    from llama_index.retrievers.bm25 import BM25Retriever

    if top_k <= 0 or not nodes:
        raise ValueError("BM25 requires nodes and a top_k greater than zero.")
    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=min(top_k, len(nodes)),
        token_pattern=BM25_TOKEN_PATTERN,
        language="en",
    )


def build_fusion_retriever(vector_retriever, bm25_retriever, llm, top_k: int):
    """Fuse ranks rather than combining incompatible raw relevance scores."""
    if top_k <= 0:
        raise ValueError("Fusion top_k must be greater than zero.")
    # Rewriting is explicit; fusion must not generate additional queries.
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        llm=llm,
        mode="reciprocal_rerank",
        similarity_top_k=top_k,
        num_queries=1,
        use_async=False,
    )


def build_policy_retriever(index, llm, category: str | None = None):
    """Apply scope before both searches; never broaden an empty category."""
    filters = build_metadata_filters(category)
    nodes = select_policy_nodes(load_index_nodes(index), category)
    if not nodes:
        raise ValueError(
            f"No indexed policies match category {category!r}. Build a new KB_NAME "
            "with business metadata before using this scope."
        )

    candidate_count = min(CANDIDATE_TOP_K, len(nodes))
    vector_retriever = index.as_retriever(
        similarity_top_k=candidate_count,
        filters=filters,
    )
    bm25_retriever = build_bm25_retriever(nodes, top_k=candidate_count)
    fusion_retriever = build_fusion_retriever(
        vector_retriever,
        bm25_retriever,
        llm,
        top_k=min(FUSION_TOP_K, len(nodes)),
    )
    return QueryRewriteRetriever(fusion_retriever, llm)
