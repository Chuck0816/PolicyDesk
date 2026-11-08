"""Query preparation and hybrid retrieval for company policies."""

import logging
import re
from collections import Counter

from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from prompts import QUERY_REWRITE_PROMPT
from llama_index.core.vector_stores import (
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)

from ingestion import POLICY_CATEGORIES


LOGGER = logging.getLogger(__name__)
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")

CANDIDATE_TOP_K = 8
FUSION_TOP_K = 10


class QueryRewriteRetriever(BaseRetriever):
    """Prepare retrieval text without changing the synthesis question."""

    '''
    Arguments passed to __init__() must be stored as self.xxx 
    so other methods in the same object can access them.
    '''
    def __init__(self, base_retriever: BaseRetriever, llm):
        super().__init__(callback_manager=base_retriever.callback_manager)
        self._base_retriever = base_retriever
        self._llm = llm


    def _predict_rewrite(self, original: str) -> str:
        """Return one rewritten query produced by the supplied LLM.

        TODO 1:
        - Call self._llm.predict with QUERY_REWRITE_PROMPT.
        - Supply original through the prompt's "query" variable.
        - Return the generated text.
        - Do not answer the question or generate multiple queries.

        Validation and fallback are handled by _rewrite().
        """
        return self._llm.predict(
            QUERY_REWRITE_PROMPT, 
            query = original,
            ).strip()


    def _rewrite(self, original: str) -> str:
        try:
            rewritten = self._predict_rewrite(original).strip()
        except NotImplementedError:
            # Unfinished exercises must remain visible.
            raise
        except Exception:
            LOGGER.warning(
                "Query rewrite failed; using the original question."
            )
            return original

        
        # `Counter()` returns a dictionary-like object that records how many times each number appears, 
        # such as `{'1': 5, '2': 3}`. 
       # We use it to ensure the rewritten query preserves all the numbers from the original question.
        original_numbers = Counter(NUMBER_PATTERN.findall(original))
        rewritten_numbers = Counter(NUMBER_PATTERN.findall(rewritten))

        if (
            not rewritten
            or "\n" in rewritten
            or len(rewritten) > 2000
            or original_numbers != rewritten_numbers
        ):
            return original

        return rewritten
    
    """Create the course-style AND filters for an optional policy scope."""
    def build_metadata_filters(category=None):
        if category is None:
            return None
        if category not in POLICY_CATEGORIES.values():
            raise ValueError(f"Unknown policy category: {category}")
        return MetadataFilters(
            filters=[
                MetadataFilter(key="category", value=category),
                MetadataFilter(key="doc_type", value="company_policy"),
            ],
            condition=FilterCondition.AND,
        )

    def select_policy_nodes(nodes, category=None):
        """Restrict the BM25 corpus to the same scope as vector retrieval."""
        if category is None:
            return nodes

        return [
            node
            for node in nodes
            if node.metadata.get("category") == category
            and node.metadata.get("doc_type") == "company_policy"
        ]

    # QueryBundle wraps the query; 
    # query_bundle.query_str contains the query text.
    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> list[NodeWithScore]:
        original = query_bundle.query_str.strip()
        if not original:
            raise ValueError("The question must not be empty.")

        rewritten = self._rewrite(original)

        # The original QueryBundle remains available to the query engine.
        
        if rewritten == original:
            return self._base_retriever.retrieve(query_bundle) # _base_retriever decides which retrival model to use

        nodes = self._base_retriever.retrieve(rewritten)
        if nodes:
            return nodes

        return self._base_retriever.retrieve(query_bundle)


    def load_index_nodes(index):
        """Load exactly the nodes referenced by the saved vector index.
        index.index_struct.nodes_dict.values() retrieve the node by id
        dict.fromkeys() delete the duplicate
        """
        node_ids = list(dict.fromkeys(index.index_struct.nodes_dict.values()))

        if not node_ids:
            raise ValueError(
                "The index has no saved node references. Build a new knowledge "
                "base with store_nodes_override=True."
            )

        return index.docstore.get_nodes(node_ids)


    def build_bm25_retriever(nodes, top_k: int):
        """Build keyword retrieval over the existing indexed nodes.

        TODO 2:
        - Use BM25Retriever.from_defaults.
        - Supply nodes rather than reading or splitting the files again.
        - Return at most top_k results.
        - Use English tokenization for the policy corpus.
        - Preserve single-character numeric tokens with:
        token_pattern=r"(?u)\\b\\w+\\b"

        Do not persist a separate local corpus.
        """
        from llama_index.retrievers.bm25 import BM25Retriever

        if top_k < 0:
            raise ValueError("top_k must be greater than zero.")

        # nodes can be directly passed into BM25Retriever
        bm25_retriever = BM25Retriever.from_defaults(
            nodes = nodes,
            similarity_top_k=top_k,
            token_pattern=r"(?u)\b\w+\b",
            language="en",
        )

        return bm25_retriever




    def build_fusion_retriever(vector_retriever, bm25_retriever, llm, top_k: int):
        """Fuse semantic and keyword results using reciprocal rank fusion.

        TODO 3:
        - Construct and return QueryFusionRetriever.
        - Include both supplied retrievers.
        - Pass the supplied llm explicitly.
        - Use mode="reciprocal_rerank".
        - Set similarity_top_k to top_k.
        - Set num_queries=1 to disable additional query generation.
        - Set use_async=False for the current synchronous application.

        Use rank fusion rather than adding raw BM25 and vector scores.
        """
        fusion_retriever = QueryFusionRetriever(
            retrievers=[
                vector_retriever,
                bm25_retriever,
            ],
            llm = llm, # llm used to generate other queries
            mode="reciprocal_rerank", # RRF ranking
            similarity_top_k=top_k,
            num_queries=1, # whether rewrite to multiple queries. 
            use_async=False,
        )
        return fusion_retriever





    def build_policy_retriever(index, llm):
        """Assemble query rewriting around the hybrid retrieval pipeline."""
        nodes = load_index_nodes(index)
        candidate_count = min(CANDIDATE_TOP_K, len(nodes))

        vector_retriever = index.as_retriever(
            similarity_top_k=candidate_count,
        )
        bm25_retriever = build_bm25_retriever(
            nodes,
            top_k=candidate_count,
        )
        fusion_retriever = build_fusion_retriever(
            vector_retriever,
            bm25_retriever,
            llm,
            top_k=min(FUSION_TOP_K, len(nodes)),
        )
        # high level wrapper
        return QueryRewriteRetriever(fusion_retriever, llm)