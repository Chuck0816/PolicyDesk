"""A local, single-question web demo using the existing PolicyDesk query service."""

import logging

import streamlit as st
from llama_index.core import load_index_from_storage
from llama_index.core.schema import MetadataMode

import services
import storage
from base_llm import create_models
from ingestion import DATA_DIR

EXAMPLES = {
    "Leave approvals": "Who approves four working days of leave?",
    "Travel expenses": "What is the hotel limit, and when must I submit my travel claim?",
    "Policy boundaries": "How many days of annual leave am I entitled to?",
}


@st.cache_resource(show_spinner=False)
def load_index_and_models(knowledge_base):
    """Load shared read-only resources on the first question, keyed by namespace."""
    context, _ = storage.connect_storage(services)
    if not context.index_store.get_index_struct(storage.INDEX_ID):
        raise ValueError(f"Build the {knowledge_base} knowledge base before asking questions.")
    llm, embed_model = create_models()
    services.configure_settings(llm, embed_model)
    index = load_index_from_storage(context, index_id=storage.INDEX_ID)
    return index, llm


def source_details(response):
    """Snapshot retrieved text and scores for display across Streamlit reruns."""
    return [
        {
            "name": node.metadata.get("file_name", "Unknown source"),
            "score": node.score,
            "text": node.get_content(metadata_mode=MetadataMode.NONE),
        }
        for node in response.source_nodes
    ]


def render_sources(sources):
    st.divider()
    st.markdown("**Retrieved policy excerpts**")
    st.caption("Similarity indicates retrieval relevance, not answer confidence.")
    if not sources:
        st.info("No source excerpts were returned.")
    for number, source in enumerate(sources, start=1):
        with st.expander(f"{number:02d}  /  {source['name']}"):
            if source["score"] is not None:
                st.caption(f"Similarity: {source['score']:.4f}")
            st.text(source["text"])


def render_sidebar():
    with st.sidebar:
        st.markdown("## PolicyDesk")
        st.caption("YOUR COMPANY POLICY ASSISTANT")
        st.divider()
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
        st.caption("Basic RAG demo · Local access")


def main():
    st.set_page_config(page_title="PolicyDesk | Policy assistant", page_icon="📘", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stAppDeployButton"] {display: none;}
        .block-container {max-width: 1080px; padding-top: 3rem; padding-bottom: 2rem;}
        h1 {letter-spacing: -0.045em; font-weight: 750; line-height: 1.12;}
        [data-testid="stSidebar"] {border-right: 1px solid #e0e5ed;}
        [data-testid="stForm"] {background: white; border-radius: 16px; padding: 1.5rem;}
        [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px;}
        .eyebrow {color: #315dcc; font-size: .78rem; letter-spacing: .14em;
                  font-weight: 700; margin-bottom: .8rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_sidebar()
    st.markdown(
        '<div class="eyebrow">EMPLOYEE SUPPORT / POLICY SEARCH</div>', unsafe_allow_html=True
    )
    st.title("Company policies. Clear answers.")
    st.markdown("Ask about leave, travel, or approvals. Get an answer grounded in your policies.")
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
                        index, llm = load_index_and_models(storage.KB_NAME)
                        engine = services.build_query_engine(index, llm)
                        response = engine.query(question)
                    answer = st.write_stream(response.response_gen)
                    sources = source_details(response)
                    render_sources(sources)
                    st.session_state["result"] = {
                        "question": question, "answer": answer, "sources": sources
                    }
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
