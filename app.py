"""A local, single-question web demo using the existing PolicyDesk query service."""

import logging

import streamlit as st
from llama_index.core import load_index_from_storage
from llama_index.core.schema import MetadataMode

import services
import storage
from base_llm import create_models, create_reranker
from ingestion import DATA_DIR, POLICY_CATEGORIES

EXAMPLES = {
    "Leave approvals": "Who approves four working days of leave?",
    "Travel expenses": "What is the hotel limit, and when must I submit my travel claim?",
    "Policy boundaries": "How many days of annual leave am I entitled to?",
}


@st.cache_resource(show_spinner=False)
def load_index_and_models(knowledge_base):
    """Share the saved index and model resources across category-specific engines."""
    context, _ = storage.connect_storage(services)

    if not context.index_store.get_index_struct(storage.INDEX_ID):
        raise ValueError(f"Build the {knowledge_base} knowledge base before asking questions.")

    llm, embed_model = create_models()
    services.configure_settings(llm, embed_model)

    index = load_index_from_storage(
        context,
        index_id=storage.INDEX_ID,
    )
    reranker = create_reranker()

    return index, llm, reranker


@st.cache_resource(show_spinner=False)
def load_query_engine(knowledge_base, category=None):
    """Cache each scope while reusing one set of model weights."""
    index, llm, reranker = load_index_and_models(knowledge_base)
    return services.build_query_engine(
        index, llm, advanced=True, reranker=reranker, category=category
    )


def source_details(response):
    """Snapshot retrieved text and scores for display across Streamlit reruns."""
    return [
        {
            "name": node.metadata.get("file_name", "Unknown source"),
            "policy_id": node.metadata.get("policy_id", "Unknown policy"),
            "version": node.metadata.get("version", "Unknown version"),
            "category": node.metadata.get("category", "Unspecified"),
            "score": node.score,
            "text": node.get_content(metadata_mode=MetadataMode.NONE),
        }
        for node in response.source_nodes
    ]


def render_sources(sources):
    st.divider()
    st.markdown("**Retrieved policy excerpts**")
    st.caption("Reranker scores indicate relative relevance, not answer confidence.")

    if not sources:
        st.info("No source excerpts were returned.")

    for number, source in enumerate(sources, start=1):
        with st.expander(f"{number:02d}  /  {source['name']}"):
            st.caption(
                f"{source['policy_id']} · Version {source['version']} · "
                f"{source['category'].replace('_', ' ').title()}"
            )
            if source["score"] is not None:
                st.caption(f"Reranker score: {source['score']:.4f}")
            st.text(source["text"])


def clear_result():
    """Discard evidence from a previous search scope."""
    st.session_state.pop("result", None)


def render_sidebar():
    with st.sidebar:
        st.markdown("## PolicyDesk")
        st.caption("YOUR COMPANY POLICY ASSISTANT")
        st.divider()
        category = st.selectbox(
            "Search scope",
            options=[None, *sorted(set(POLICY_CATEGORIES.values()))],
            format_func=lambda value: (
                "All policies" if value is None else value.replace("_", " ").title()
            ),
            key="policy_scope",
            on_change=clear_result,
        )
        st.caption("Both semantic and keyword search stay within this scope.")
        st.markdown("**Policy library**")
        st.caption("Local source files; indexed content may differ until rebuilt.")
        for path in sorted(DATA_DIR.glob("*.txt")):
            with st.expander(path.stem.replace("_", " ").title()):
                st.text(path.read_text(encoding="utf-8"))
        st.divider()
        st.markdown("**How it works**")
        st.caption("01 / Ask a question about your company policies.")
        st.caption("02 / Read an answer based on relevant policy excerpts.")
        st.caption("03 / Open the sources to check the original wording.")
        st.divider()
        st.caption("Policy search · Local access")
    return category


def main():
    st.set_page_config(page_title="PolicyDesk | Policy assistant", page_icon="📘", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stAppDeployButton"] {display: none;}
        .block-container {max-width: 1080px; padding-top: 3rem; padding-bottom: 2rem;}
        h1 {letter-spacing: -0.045em; font-weight: 750; line-height: 1.12;}
        [data-testid="stForm"] {border-radius: 16px; padding: 1.5rem;}
        [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px;}
        .eyebrow {font-size: .78rem; letter-spacing: .14em;
                  font-weight: 700; margin-bottom: .8rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    category = render_sidebar()
    st.markdown(
        '<div class="eyebrow">EMPLOYEE SUPPORT / POLICY SEARCH</div>', unsafe_allow_html=True
    )
    st.title("Company policies. Clear answers.")
    st.markdown("Ask about company policies. Get an answer grounded in the selected scope.")
    st.caption("Each question is independent and uses the indexed company policies.")
    st.write("")
    st.markdown("**Try a question**")
    for column, (label, question) in zip(st.columns(3), EXAMPLES.items()):
        if column.button(label, use_container_width=True):
            st.session_state["question"] = question

    with st.form("policy_question"):
        st.text_area(
            "What would you like to know?",
            key="question",
            placeholder="For example: Who approves four working days of leave?",
            height=100,
            max_chars=2000,
        )
        submitted = st.form_submit_button("Ask PolicyDesk →", type="primary")

    if submitted:
        question = st.session_state["question"].strip()
        if not question:
            st.warning("Enter a question before submitting.")
        else:
            st.session_state.pop("result", None)
            answer_area = st.empty()
            try:
                with answer_area.container(border=True):
                    st.caption("ANSWER")
                    with st.spinner("Looking up the policy and preparing your answer..."):
                        engine = load_query_engine(storage.KB_NAME, category)
                        response = engine.query(question)
                    answer = st.write_stream(response.response_gen)
                    sources = source_details(response)
                    render_sources(sources)
                    st.session_state["result"] = {
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                        "category": category,
                    }
            except ValueError as error:
                answer_area.empty()
                logging.getLogger(__name__).exception("Policy configuration or scope failed")
                st.error(f"Could not complete the question. {error}")
            except Exception:
                answer_area.empty()
                logging.getLogger(__name__).exception("Policy query failed")
                st.error(
                    "Could not complete the question. Check the database services, API key, "
                    "and saved index. See the terminal for details, then retry."
                )
    elif result := st.session_state.get("result"):
        with st.container(border=True):
            st.caption("ANSWER TO YOUR LAST QUESTION")
            st.markdown(f"**{result['question']}**")
            st.markdown(result["answer"])
            render_sources(result["sources"])

    st.caption(
        "Answers use retrieved policy context. Contact HR or Finance for unresolved questions."
    )


if __name__ == "__main__":
    main()
