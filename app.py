"""A local web demo: add or delete policy PDFs, then ask questions about them."""

import logging
import tempfile
from pathlib import Path

import streamlit as st
from llama_index.core import load_index_from_storage
from llama_index.core.schema import MetadataMode

import pdf_ingestion
import services
import storage
from base_llm import create_models, create_reranker

LOGGER = logging.getLogger(__name__)
CONTENT_LABELS = {"text": "Text", "table": "Table", "figure": "Figure"}


@st.cache_resource(show_spinner=False)
def load_models():
    """Load Qwen, the embedding model, and the reranker once per process."""
    llm, embed_model = create_models()
    services.configure_settings(llm, embed_model)
    return llm, create_reranker()


@st.cache_resource(show_spinner=False)
def load_query_engine(knowledge_base):
    """Cache one engine until a PDF is added or deleted; BM25 uses this snapshot."""
    context, _ = storage.connect_storage(services)
    if not context.index_store.get_index_struct(storage.INDEX_ID):
        raise ValueError(f"Add a PDF to {knowledge_base} before asking questions.")
    llm, reranker = load_models()
    index = load_index_from_storage(context, index_id=storage.INDEX_ID)
    return services.build_query_engine(index, llm, advanced=True, reranker=reranker)


@st.cache_resource(show_spinner=False)
def connect_docstore(knowledge_base):
    """Reuse one MongoDB client for the document list."""
    return storage.connect_docstore()


def list_documents():
    return pdf_ingestion.list_pdfs(connect_docstore(storage.KB_NAME))


def notify(kind, message):
    """Keep a message across the rerun that follows a change."""
    st.session_state.setdefault("notices", []).append((kind, message))


def refresh_after_change():
    """Rebuild the engine on the next question so both search branches see the change."""
    load_query_engine.clear()
    st.session_state.pop("result", None)


def ingest_upload(name, data):
    """Save the upload under its own name so sources show it, then append it."""
    load_models()  # Configures the embedding model used for indexing.
    context, collection = storage.connect_storage(services, create=True)
    with tempfile.TemporaryDirectory() as folder:
        pdf = Path(folder) / Path(name).name
        pdf.write_bytes(data)
        return pdf_ingestion.ingest_pdf(pdf, context, collection, work_dir=Path(folder) / "mineru")


def skipped_message(skipped):
    lines = []
    for item in skipped:
        caption = f" ({item['caption']})" if item["caption"] else ""
        lines.append(f"- Page {item['page']}{caption}: {item['reason'][:200]}")
    return (
        f"Skipped {len(skipped)} figure(s) that could not be described. Their content is "
        "not searchable; the rest of the PDF was added.\n" + "\n".join(lines)
    )


def add_upload(upload):
    with st.status(f"Adding {upload.name}...", expanded=True) as status:
        st.write("Parsing the PDF and describing figures. Large files can take minutes.")
        try:
            doc_id, count, skipped = ingest_upload(upload.name, upload.getvalue())
        except Exception as error:
            LOGGER.exception("PDF ingestion failed")
            status.update(label=f"Could not add {upload.name}", state="error")
            st.error(str(error))
            return
    refresh_after_change()
    notify("success", f"Added {upload.name} ({count} chunks).")
    if skipped:
        notify("warning", skipped_message(skipped))
    # A new uploader key clears the selected file after a successful add.
    st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1
    st.rerun()


def request_delete(doc_id):
    st.session_state["pending_delete"] = doc_id


def cancel_delete():
    st.session_state.pop("pending_delete", None)


def delete_document(doc_id, file_name):
    cancel_delete()
    try:
        load_models()  # Loading the index for deletion resolves the embedding model.
        context, _ = storage.connect_storage(services)
        pdf_ingestion.delete_pdf(context, doc_id)
    except Exception as error:
        LOGGER.exception("PDF deletion failed")
        notify("error", f"Could not delete {file_name}. {error}")
        return
    refresh_after_change()
    notify("success", f"Deleted {file_name}.")


def render_document(document):
    doc_id, name = document["doc_id"], document["file_name"]
    with st.container(border=True):
        st.markdown(f"**{name}**")
        details = [f"{document['chunks']} chunks"]
        if document["uploaded_at"]:
            details.append(f"added {document['uploaded_at'][:16].replace('T', ' ')} UTC")
        if document["skipped_figures"]:
            details.append(f"{document['skipped_figures']} figure(s) skipped")
        st.caption(" · ".join(details))
        if st.session_state.get("pending_delete") == doc_id:
            st.caption("Delete this PDF from the knowledge base?")
            confirm, cancel = st.columns(2)
            confirm.button(
                "Delete",
                key=f"confirm-{doc_id}",
                type="primary",
                on_click=delete_document,
                args=(doc_id, name),
            )
            cancel.button("Cancel", key=f"cancel-{doc_id}", on_click=cancel_delete)
        else:
            st.button("Delete...", key=f"delete-{doc_id}", on_click=request_delete, args=(doc_id,))


def render_sidebar(documents):
    with st.sidebar:
        st.markdown("## PolicyDesk")
        st.caption("YOUR COMPANY POLICY ASSISTANT")
        st.divider()
        for kind, message in st.session_state.pop("notices", []):
            getattr(st, kind)(message)
        st.markdown("**Add a policy PDF**")
        upload = st.file_uploader(
            "Policy PDF",
            type=["pdf"],
            key=f"upload-{st.session_state.get('upload_round', 0)}",
            label_visibility="collapsed",
        )
        if st.button("Add to knowledge base", disabled=upload is None, width="stretch"):
            add_upload(upload)
        st.divider()
        if documents is None:
            st.caption("The document list is unavailable.")
        else:
            st.markdown(f"**Documents ({len(documents)})**")
            if not documents:
                st.caption("No PDFs yet.")
            for document in documents:
                render_document(document)
        st.divider()
        st.markdown("**How it works**")
        st.caption("01 / Add your company's policy PDFs.")
        st.caption("02 / Ask a question in Chinese or English.")
        st.caption("03 / Open the sources to check the original page.")


def source_details(response):
    """Snapshot retrieved text and scores for display across Streamlit reruns."""
    return [
        {
            "name": node.metadata.get("file_name", "Unknown source"),
            "page": node.metadata.get("page"),
            "kind": CONTENT_LABELS.get(node.metadata.get("content_type"), "Text"),
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
            page = f"Page {source['page']} · " if source["page"] is not None else ""
            st.caption(f"{page}{source['kind']}")
            if source["score"] is not None:
                st.caption(f"Reranker score: {source['score']:.4f}")
            st.text(source["text"])


def answer(question):
    st.session_state.pop("result", None)
    answer_area = st.empty()
    try:
        with answer_area.container(border=True):
            st.caption("ANSWER")
            with st.spinner("Looking up your policies and preparing an answer..."):
                engine = load_query_engine(storage.KB_NAME)
                response = engine.query(question)
            text = st.write_stream(response.response_gen)
            sources = source_details(response)
            render_sources(sources)
            st.session_state["result"] = {"question": question, "answer": text, "sources": sources}
    except ValueError as error:
        answer_area.empty()
        LOGGER.exception("Policy configuration failed")
        st.error(f"Could not complete the question. {error}")
    except Exception:
        answer_area.empty()
        LOGGER.exception("Policy query failed")
        st.error(
            "Could not complete the question. Check the database services, API key, "
            "and saved index. See the terminal for details, then retry."
        )


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
    try:
        documents = list_documents()
    except Exception:
        LOGGER.exception("Could not read the document list")
        documents = None
        st.error("Could not reach the database services. Start them with Docker, then reload.")
    render_sidebar(documents)
    st.markdown(
        '<div class="eyebrow">EMPLOYEE SUPPORT / POLICY SEARCH</div>', unsafe_allow_html=True
    )
    st.title("Company policies. Clear answers.")
    st.markdown("Ask about your company's policy PDFs in Chinese or English.")
    st.caption("Each question is independent. Answers follow the language of your question.")
    if documents == []:
        st.info("Add a policy PDF in the sidebar to start asking questions.")

    with st.form("policy_question"):
        st.text_area(
            "What would you like to know?",
            key="question",
            placeholder="For example: 出差住宿标准是多少？ / Who approves four days of leave?",
            height=100,
            max_chars=2000,
        )
        submitted = st.form_submit_button(
            "Ask PolicyDesk →", type="primary", disabled=not documents
        )

    if submitted:
        question = st.session_state["question"].strip()
        if not question:
            st.warning("Enter a question before submitting.")
        else:
            answer(question)
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
