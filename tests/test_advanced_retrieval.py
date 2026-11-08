"""Offline behavior checks for query rewriting and hybrid policy retrieval."""

from unittest.mock import Mock

import pytest
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from retrieval import (
    QueryRewriteRetriever,
    build_bm25_retriever,
    build_fusion_retriever,
)


class FixedRetriever(BaseRetriever):
    """Return known nodes and record the actual retrieval queries."""

    def __init__(self, nodes):
        super().__init__()
        self.nodes = nodes
        self.queries = []

    def _retrieve(self, query_bundle):
        self.queries.append(query_bundle.query_str)
        return list(self.nodes)


def test_rewrite_changes_retrieval_text_without_mutating_original():
    evidence = NodeWithScore(
        node=TextNode(text="Extended leave requires department head approval."),
        score=0.8,
    )
    base = FixedRetriever([evidence])
    llm = Mock()
    llm.predict.return_value = "Who approves 4 working days of leave?"

    bundle = QueryBundle("请假4个工作日需要谁批准？")
    result = QueryRewriteRetriever(base, llm).retrieve(bundle)

    assert result
    assert base.queries == ["Who approves 4 working days of leave?"]
    assert bundle.query_str == "请假4个工作日需要谁批准？"
    llm.predict.assert_called_once()


@pytest.mark.parametrize(
    "rewritten",
    [
        "Who approves 3 working days of leave?",
        "",
        "First question\nSecond question",
    ],
)
def test_invalid_rewrite_uses_original_question(rewritten):
    base = FixedRetriever([])
    llm = Mock()
    llm.predict.return_value = rewritten
    original = "Who approves 4 working days of leave?"

    QueryRewriteRetriever(base, llm).retrieve(original)

    assert base.queries == [original]


def test_provider_failure_uses_original_question():
    base = FixedRetriever([])
    llm = Mock()
    llm.predict.side_effect = RuntimeError("Provider unavailable")

    QueryRewriteRetriever(base, llm).retrieve("Who approves leave?")

    assert base.queries == ["Who approves leave?"]


def test_fusion_preserves_evidence_found_only_by_keyword_search():
    salary = TextNode(
        id_="salary",
        text="Payroll pays monthly salary on the twenty-fifth.",
    )
    travel = TextNode(
        id_="travel",
        text="Business travel accommodation requires approval.",
    )

    # Simulate a semantic search that misses the salary passage.
    dense = FixedRetriever([NodeWithScore(node=travel, score=0.9)])
    sparse = build_bm25_retriever([salary, travel], top_k=1)

    fusion = build_fusion_retriever(
        dense,
        sparse,
        MockLLM(),
        top_k=2,
    )
    results = fusion.retrieve("payroll")
    ids = [item.node.node_id for item in results]

    assert set(ids) == {"salary", "travel"}
    assert len(ids) == len(set(ids))
    assert dense.queries == ["payroll"]


@pytest.mark.parametrize("top_k", [0, -1])
def test_bm25_rejects_nonpositive_top_k(top_k):
    with pytest.raises(ValueError, match="greater than zero"):
        build_bm25_retriever([TextNode(text="Leave rules.")], top_k)


def test_bm25_rejects_empty_corpus():
    with pytest.raises(ValueError, match="requires nodes"):
        build_bm25_retriever([], 1)


def test_empty_rewrite_results_retry_original_with_the_same_retriever():
    base = FixedRetriever([])
    llm = Mock()
    llm.predict.return_value = "Leave approval"
    assert QueryRewriteRetriever(base, llm).retrieve("Who approves leave?") == []
    assert base.queries == ["Leave approval", "Who approves leave?"]


def test_empty_question_is_rejected_before_rewriting():
    llm = Mock()
    with pytest.raises(ValueError, match="must not be empty"):
        QueryRewriteRetriever(FixedRetriever([]), llm).retrieve("   ")
    llm.predict.assert_not_called()
