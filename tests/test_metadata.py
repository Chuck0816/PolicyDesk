"""Business-header validation and scope isolation across retrieval branches."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import FilterCondition

import services
from ingestion import POLICY_CATEGORIES, load_documents, parse_policy_header
from retrieval import (
    QueryRewriteRetriever,
    build_metadata_filters,
    build_policy_retriever,
    select_policy_nodes,
)


@pytest.fixture
def policy_nodes():
    return services.split_documents(load_documents())


def test_policy_headers_and_metadata_survive_splitting(policy_nodes):
    required = {"policy_title", "policy_id", "version", "owner", "doc_type", "language", "category"}
    assert len(load_documents()) == len(POLICY_CATEGORIES) + 1
    for node in policy_nodes:
        assert required <= node.metadata.keys()
        assert all(node.metadata[key] for key in required)
        assert node.metadata["language"] == "en"
        if node.metadata["file_name"] in POLICY_CATEGORIES:
            assert node.metadata["category"] == POLICY_CATEGORIES[node.metadata["file_name"]]
            assert node.metadata["doc_type"] == "company_policy"
    salary = next(doc for doc in load_documents() if doc.metadata["category"] == "salary")
    assert salary.metadata["owner"] == "Human Resources (HR) and Finance"


@pytest.mark.parametrize(
    "header",
    [
        "",
        "Policy title\nPolicy ID: X\n",
        "Title\nVersion: 1.0\n",
        "Title\nPolicy ID: HR-001\nVersion: 1.0\n",
        "Title\nPolicy ID: HR-001\nVersion: 1.0\nPolicy owner:   \n",
    ],
)
def test_missing_business_header_is_rejected(header):
    with pytest.raises(ValueError):
        parse_policy_header(header)


def test_new_policy_file_requires_explicit_category(tmp_path, monkeypatch):
    (tmp_path / "new_policy.txt").write_text(
        "New Policy\nPolicy ID: NEW-001\nVersion: 1.0\nPolicy owner: HR\n\nRules.",
        encoding="utf-8",
    )
    monkeypatch.setattr("ingestion.DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="Register"):
        load_documents()


@pytest.mark.parametrize("category", sorted(set(POLICY_CATEGORIES.values())))
def test_scopes_select_only_the_matching_company_policy(policy_nodes, category):
    scoped = select_policy_nodes(policy_nodes, category)
    assert scoped
    assert {node.metadata["category"] for node in scoped} == {category}
    assert {node.metadata["doc_type"] for node in scoped} == {"company_policy"}
    assert {node.metadata["file_name"] for node in scoped} == {
        name for name, value in POLICY_CATEGORIES.items() if value == category
    }
    filters = build_metadata_filters(category)
    assert filters.condition == FilterCondition.AND
    assert {(item.key, item.value) for item in filters.filters} == {
        ("category", category),
        ("doc_type", "company_policy"),
    }


def test_all_policies_include_general_guidance(policy_nodes):
    assert build_metadata_filters() is None
    assert select_policy_nodes(policy_nodes) is policy_nodes
    assert any(node.metadata["file_name"] == "manual_faq" for node in policy_nodes)


def test_general_guidance_cannot_enter_a_category_by_category_alone():
    guidance = TextNode(
        text="Ask HR.",
        metadata={"category": "leave", "doc_type": "general_guidance"},
    )
    assert select_policy_nodes([guidance], "leave") == []


@pytest.mark.parametrize("category", ["", "unknown", "general"])
def test_invalid_scope_is_rejected(category):
    with pytest.raises(ValueError, match="Unknown policy category"):
        build_metadata_filters(category)
    with pytest.raises(ValueError):
        select_policy_nodes([], category)


def test_old_index_without_business_metadata_fails_before_search():
    node = TextNode(id_="legacy", text="Managers approve leave.")
    index = SimpleNamespace(
        index_struct=SimpleNamespace(nodes_dict={"legacy": "legacy"}),
        docstore=Mock(get_nodes=Mock(return_value=[node])),
        as_retriever=Mock(),
    )
    with pytest.raises(ValueError, match="business metadata"):
        build_policy_retriever(index, MockLLM(), category="leave")
    index.as_retriever.assert_not_called()


class RecordingPostprocessor(BaseNodePostprocessor):
    """Record the synthesis question while leaving evidence unchanged."""

    seen_questions: list[str] = []

    def _postprocess_nodes(self, nodes, query_bundle=None):
        self.seen_questions.append(query_bundle.query_str)
        return nodes


def test_hybrid_scope_and_original_synthesis_question(monkeypatch):
    Settings.embed_model = MockEmbedding(embed_dim=8)
    Settings.llm = MockLLM(max_tokens=16)
    nodes = [
        TextNode(
            text="Managers approve leave requests.",
            metadata={"category": "leave", "doc_type": "company_policy"},
        ),
        TextNode(
            text="Payroll pays salary monthly.",
            metadata={"category": "salary", "doc_type": "company_policy"},
        ),
        TextNode(
            text="General guidance: ask HR about salary and leave.",
            metadata={"category": "general", "doc_type": "general_guidance"},
        ),
    ]
    index = VectorStoreIndex(nodes, store_nodes_override=True)
    monkeypatch.setattr(QueryRewriteRetriever, "_predict_rewrite", lambda self, text: "payroll")
    postprocessor = RecordingPostprocessor()
    engine = services.build_query_engine(
        index, Settings.llm, advanced=True, reranker=postprocessor, category="leave"
    )
    response = engine.query("Who approves leave?")
    assert "".join(response.response_gen)
    assert response.source_nodes
    assert {item.metadata["category"] for item in response.source_nodes} == {"leave"}
    assert postprocessor.seen_questions == ["Who approves leave?"]


def test_baseline_respects_the_same_scope():
    Settings.embed_model = MockEmbedding(embed_dim=8)
    Settings.llm = MockLLM(max_tokens=16)
    index = VectorStoreIndex(
        [
            TextNode(text="Leave.", metadata={"category": "leave", "doc_type": "company_policy"}),
            TextNode(text="Salary.", metadata={"category": "salary", "doc_type": "company_policy"}),
        ]
    )
    response = services.build_query_engine(index, Settings.llm, category="salary").query("leave")
    assert response.source_nodes
    assert {item.metadata["category"] for item in response.source_nodes} == {"salary"}
