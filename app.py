"""Web demo: ask about company policy PDFs, and add or delete them."""

import html
import logging
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import streamlit as st
from llama_index.core import load_index_from_storage
from llama_index.core.schema import MetadataMode

import pdf_ingestion
import services
import storage
from base_llm import create_models, create_reranker
from ui_text import DEFAULT_LANGUAGE, LANGUAGES, translate

LOGGER = logging.getLogger(__name__)
KIND_KEYS = {"text": "kind_text", "table": "kind_table", "figure": "kind_figure"}
UPLOAD_STEPS = ("upload", "parse", "describe", "index")
TABLE_COLUMNS = [4, 2.2, 1, 2, 1.2]

# Translucent colors read well on both of Streamlit's light and dark themes.
STYLE = """
<style>
:root {
  --pd-line: rgba(128,128,128,.26); --pd-line-2: rgba(128,128,128,.15);
  --pd-surface: rgba(128,128,128,.07);
  --pd-accent: #1fa595; --pd-accent-soft: rgba(31,165,149,.15);
  --pd-ok: #23a26d; --pd-ok-soft: rgba(35,162,109,.15);
  --pd-warn: #c98a12; --pd-warn-soft: rgba(233,170,40,.15);
  --pd-danger: #e0475a; --pd-danger-soft: rgba(224,71,90,.12);
}
[data-testid="stAppDeployButton"] {display: none;}
header[data-testid="stHeader"] {background: transparent;}
/* Start below Streamlit's fixed toolbar, which otherwise covers the navigation. */
.block-container {max-width: MAX_WIDTH; padding-top: 3.6rem; padding-bottom: 3rem;}
.pd-muted {color: color-mix(in srgb, currentColor 62%, transparent);}
.pd-brand {display: flex; align-items: center; gap: .6rem; font-weight: 700; font-size: 1.05rem;}
.pd-mark {width: 22px; height: 22px; border-radius: 50%; border: 2px solid var(--pd-accent);
  display: grid; place-items: center;}
.pd-mark::after {content: ""; width: 8px; height: 8px; border-radius: 50%;
  background: var(--pd-accent);}
.pd-h1 {font-size: 2.1rem; font-weight: 650; letter-spacing: -.03em; margin: 1.4rem 0 .6rem;}
.pd-h2 {font-size: 1.6rem; font-weight: 650; letter-spacing: -.02em; margin: 1rem 0 .1rem;}
.pd-label {font-size: .74rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
  margin: 1.2rem 0 .5rem;}
[data-testid="stForm"] {border-radius: 18px; border: 1px solid var(--pd-line);
  box-shadow: 0 0 0 4px var(--pd-accent-soft);}
.pd-phases {display: flex; flex-wrap: wrap; gap: .4rem 1.1rem; font-size: .84rem;
  margin: .9rem 0 .2rem;}
.pd-phase {display: inline-flex; align-items: center; gap: .4rem;}
.pd-phase::before {content: ""; width: 8px; height: 8px; border-radius: 50%;
  border: 2px solid var(--pd-line);}
.pd-phase.done::before {background: var(--pd-accent); border-color: var(--pd-accent);}
.pd-phase.now::before {border-color: var(--pd-accent); animation: pd-pulse 1s infinite alternate;}
@keyframes pd-pulse {to {opacity: .3;}}
.pd-cards {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .6rem;}
.pd-card {border: 1px solid var(--pd-line); background: var(--pd-surface); border-radius: 12px;
  padding: .65rem .75rem; display: flex; flex-direction: column; gap: .45rem; min-height: 5.6rem;}
.pd-card-text {font-size: .82rem; line-height: 1.4; max-height: 2.8em; overflow: hidden;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;}
.pd-card-meta {margin-top: auto; font-size: .72rem; display: flex; gap: .35rem;
  align-items: center;}
.pd-kind {font-size: .62rem; font-weight: 700; border-radius: 4px; padding: 0 .3rem;
  white-space: nowrap; flex: none;
  background: var(--pd-accent-soft); color: var(--pd-accent);}
.st-key-answer p, .st-key-answer li {font-size: 1.06rem; line-height: 1.8;}
.pd-excerpt-head {display: flex; justify-content: space-between; font-size: .8rem;
  margin-top: .4rem;}
.pd-chip {display: inline-flex; align-items: center; gap: .35rem; font-size: .78rem;
  font-weight: 650;
  border-radius: 99px; padding: .1rem .6rem; white-space: nowrap;}
.pd-chip::before {content: ""; width: 7px; height: 7px; border-radius: 50%;
  background: currentColor;}
.pd-chip.ok {background: var(--pd-ok-soft); color: var(--pd-ok);}
.pd-chip.warn {background: var(--pd-warn-soft); color: var(--pd-warn);}
.pd-file {display: flex; gap: .7rem; align-items: center;}
.pd-pdf {width: 28px; height: 34px; border-radius: 6px; background: var(--pd-danger-soft);
  color: var(--pd-danger); font-size: .55rem; font-weight: 800; display: grid;
  place-items: end center; padding-bottom: 3px; flex: none;}
.pd-file-name {font-weight: 600; word-break: break-all;}
.pd-file-id {font-size: .72rem; font-family: ui-monospace, Menlo, monospace;}
.pd-th {font-size: .7rem; font-weight: 650; letter-spacing: .07em; text-transform: uppercase;}
.pd-num {font-variant-numeric: tabular-nums;}
[class*="st-key-row-"] {border-top: 1px solid var(--pd-line-2); padding-top: .55rem;}
[class*="st-key-confirm-"] {background: var(--pd-danger-soft); border-radius: 10px;
  padding: .6rem .8rem; color: var(--pd-danger);}
[class*="st-key-really-delete-"] button {background: var(--pd-danger);
  border-color: var(--pd-danger); color: #fff;}
.pd-mobile-only {display: none;}
.pd-notice {border-radius: 14px; padding: .75rem 1rem; font-size: .9rem; margin: .4rem 0 .6rem;
  border: 1px solid var(--pd-line);}
.pd-notice.ok {background: var(--pd-ok-soft); border-color: var(--pd-ok);}
.pd-notice.warn {background: var(--pd-warn-soft); border-color: var(--pd-warn);}
.pd-notice.error {background: var(--pd-danger-soft); border-color: var(--pd-danger);}
.pd-notice b {color: inherit;}
.pd-notice ul {margin: .35rem 0 0; padding-left: 1.1rem; font-size: .82rem;}
.pd-progress {border: 1px solid var(--pd-line); border-radius: 14px; padding: .8rem 1rem;
  margin: .4rem 0 .6rem; display: grid; gap: .5rem;}
.pd-progress-top {display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap;}
.pd-steps {display: flex; flex-wrap: wrap; gap: .3rem 1rem; font-size: .8rem;}
.pd-step.done::before {content: "✓ "; color: var(--pd-accent);}
.pd-step.now {font-weight: 650;}
.pd-step.now::before {content: "◌ "; color: var(--pd-accent);}
.pd-bar {height: 4px; border-radius: 9px; background: var(--pd-line-2); overflow: hidden;}
.pd-bar i {display: block; height: 100%; width: 30%; border-radius: 9px;
  background: var(--pd-accent);
  animation: pd-slide 1.4s ease-in-out infinite;}
@keyframes pd-slide {from {margin-left: -30%;} to {margin-left: 100%;}}
@media (max-width: 720px) {
  .pd-cards {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .pd-h1 {font-size: 1.6rem;}
}
/* Streamlit stacks columns below 640px; show the row details under the file name. */
@media (max-width: 640px) {
  .st-key-doc_head, .pd-desktop-only {display: none;}
  .pd-mobile-only {display: block;}
}
</style>
"""


def t(key, **values):
    return translate(st.session_state.get("language", DEFAULT_LANGUAGE), key, **values)


def escape(value):
    return html.escape(str(value))


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


class _PlainText(HTMLParser):
    """Turn MinerU's HTML tables into "cell | cell" lines and drop other markup."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.lines.append(" | ".join(self.row))
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        elif self.row is None:
            self.lines.extend(line.strip() for line in data.splitlines())


def plain_text(text):
    if "<" not in text:
        return text
    parser = _PlainText()
    parser.feed(text)
    parser.close()
    return "\n".join(line for line in parser.lines if line)


def snippet(text, limit=90):
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"


def kind_label(kind):
    return t(KIND_KEYS.get(kind, "kind_text"))


def source_details(response):
    """Snapshot retrieved text and scores for display across Streamlit reruns."""
    return [
        {
            "name": node.metadata.get("file_name", "Unknown source"),
            "page": node.metadata.get("page"),
            "kind": node.metadata.get("content_type", "text"),
            "score": node.score,
            "text": plain_text(node.get_content(metadata_mode=MetadataMode.NONE)),
        }
        for node in response.source_nodes
    ]


def source_location(source):
    parts = [escape(source["name"])]
    if source["page"] is not None:
        parts.append(escape(t("page", page=source["page"])))
    return " · ".join(parts)


def render_phases(slot, stage):
    """stage: "retrieve" while searching and ranking, "write" while streaming, or "done"."""
    states = {
        "retrieve": ("now", "now", ""),
        "write": ("done", "done", "now"),
        "done": ("done", "done", "done"),
    }[stage]
    spans = "".join(
        f'<span class="pd-phase {state}">{escape(t(key))}</span>'
        for key, state in zip(("phase_search", "phase_rank", "phase_write"), states)
    )
    slot.markdown(f'<div class="pd-phases pd-muted">{spans}</div>', unsafe_allow_html=True)


def render_source_cards(sources):
    st.markdown(
        f'<div class="pd-label pd-muted">{escape(t("sources_label", count=len(sources)))}</div>',
        unsafe_allow_html=True,
    )
    if not sources:
        st.info(t("no_sources"))
        return
    cards = "".join(
        '<div class="pd-card">'
        f'<div class="pd-card-text">{escape(snippet(source["text"]))}</div>'
        '<div class="pd-card-meta pd-muted">'
        f'<span class="pd-kind">{escape(kind_label(source["kind"]))}'
        f"</span>{source_location(source)}</div></div>"
        for source in sources
    )
    st.markdown(f'<div class="pd-cards">{cards}</div>', unsafe_allow_html=True)


def render_excerpts(sources):
    if not sources:
        return
    with st.expander(t("excerpts")):
        for number, source in enumerate(sources, start=1):
            score = (
                escape(t("relevance", score=f"{source['score']:.2f}"))
                if source["score"] is not None
                else ""
            )
            st.markdown(
                f'<div class="pd-excerpt-head pd-muted"><span>{number:02d} · '
                f"{source_location(source)} · {escape(kind_label(source['kind']))}</span>"
                f"<span>{score}</span></div>",
                unsafe_allow_html=True,
            )
            st.text(source["text"])


def answer(question):
    st.session_state.pop("result", None)
    area = st.empty()
    try:
        with area.container():
            phases = st.empty()
            render_phases(phases, "retrieve")
            engine = load_query_engine(storage.KB_NAME)
            response = engine.query(question)
            sources = source_details(response)
            render_source_cards(sources)
            render_phases(phases, "write")
            st.markdown(
                f'<div class="pd-label pd-muted">{escape(t("answer_label"))}</div>',
                unsafe_allow_html=True,
            )
            with st.container(key="answer"):
                text = st.write_stream(response.response_gen)
            render_phases(phases, "done")
            render_excerpts(sources)
            st.session_state["result"] = {"question": question, "answer": text, "sources": sources}
    except ValueError as error:
        area.empty()
        LOGGER.exception("Policy configuration failed")
        st.error(t("question_failed", error=error))
    except Exception:
        area.empty()
        LOGGER.exception("Policy query failed")
        st.error(t("query_failed"))


def render_saved_result(result):
    st.markdown(
        f'<div class="pd-label pd-muted">{escape(t("last_answer_label"))}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**{escape(result['question'])}**")
    render_source_cards(result["sources"])
    st.markdown(
        f'<div class="pd-label pd-muted">{escape(t("answer_label"))}</div>', unsafe_allow_html=True
    )
    with st.container(key="answer"):
        st.markdown(result["answer"])
    render_excerpts(result["sources"])


def ask_example(question):
    st.session_state["question"] = question
    st.session_state["auto_ask"] = True


def show_view(view):
    st.session_state["view"] = view


def render_ask(documents):
    st.markdown(f'<div class="pd-h1">{escape(t("ask_title"))}</div>', unsafe_allow_html=True)
    if documents == []:
        with st.container(border=True):
            st.markdown(t("empty_ask"))
            st.button(t("go_docs"), type="primary", on_click=show_view, args=("docs",))

    with st.form("policy_question"):
        st.text_area(
            t("question_label"),
            key="question",
            placeholder=t("question_placeholder"),
            height=90,
            max_chars=2000,
            label_visibility="collapsed",
        )
        scope, submit = st.columns([4, 1], vertical_alignment="center")
        scope.caption(t("scope", count=len(documents or [])))
        submitted = submit.form_submit_button(
            t("ask_button"), type="primary", disabled=not documents, width="stretch"
        )

    if submitted or st.session_state.pop("auto_ask", False):
        question = st.session_state["question"].strip()
        if not question:
            st.warning(t("enter_question"))
        else:
            answer(question)
    elif result := st.session_state.get("result"):
        render_saved_result(result)

    if documents:
        st.markdown(
            f'<div class="pd-label pd-muted">{escape(t("examples_label"))}</div>',
            unsafe_allow_html=True,
        )
        for number, example in enumerate(t("examples")):
            st.button(
                example,
                key=f"example-{number}",
                type="tertiary",
                on_click=ask_example,
                args=(example,),
            )
    st.caption(t("disclaimer"))


def notify(kind, title, body="", items=()):
    """Keep a message across the rerun that follows a change."""
    notice = {"kind": kind, "title": title, "body": body, "items": list(items)}
    st.session_state.setdefault("notices", []).append(notice)


def notice_html(notice):
    items = "".join(f"<li>{escape(item)}</li>" for item in notice["items"])
    return (
        f'<div class="pd-notice {notice["kind"]}"><b>{escape(notice["title"])}</b> '
        f"{escape(notice['body'])}{f'<ul>{items}</ul>' if items else ''}</div>"
    )


def progress_html(name, step):
    current = UPLOAD_STEPS.index(step)
    steps = "".join(
        f'<span class="pd-step {"done" if index < current else "now" if index == current else ""}">'
        f"{escape(t('step_' + key))}</span>"
        for index, key in enumerate(UPLOAD_STEPS)
    )
    return (
        '<div class="pd-progress"><div class="pd-progress-top">'
        f"<span>{escape(t('adding', name=name))}</span>"
        f'<span class="pd-muted">{escape(t("adding_hint"))}</span></div>'
        f'<div class="pd-steps pd-muted">{steps}</div><div class="pd-bar"><i></i></div></div>'
    )


def refresh_after_change():
    """Rebuild the engine on the next question so both search branches see the change."""
    load_query_engine.clear()
    st.session_state.pop("result", None)


def ingest_upload(name, data, progress=None):
    """Save the upload under its own name so sources show it, then append it."""
    load_models()  # Configures the embedding model used for indexing.
    context, collection = storage.connect_storage(services, create=True)
    with tempfile.TemporaryDirectory() as folder:
        pdf = Path(folder) / Path(name).name
        pdf.write_bytes(data)
        return pdf_ingestion.ingest_pdf(
            pdf, context, collection, work_dir=Path(folder) / "mineru", progress=progress
        )


def skipped_items(skipped):
    items = []
    for item in skipped:
        caption = f" · {item['caption']}" if item["caption"] else ""
        reason = item["reason"][:200]
        items.append(t("skipped_item", page=item["page"], caption=caption, reason=reason))
    return items


def add_upload(upload, slot):
    name = upload.name

    def show(step):
        slot.markdown(progress_html(name, step), unsafe_allow_html=True)

    show("upload")
    try:
        _, count, skipped = ingest_upload(name, upload.getvalue(), progress=show)
    except Exception as error:
        LOGGER.exception("PDF ingestion failed")
        failure = {"kind": "error", "title": t("add_failed", name=name, error=error)}
        slot.markdown(notice_html({**failure, "body": "", "items": []}), unsafe_allow_html=True)
        return
    refresh_after_change()
    if skipped:
        notify(
            "warn",
            t("added", name=name, count=count),
            t("skipped", count=len(skipped)),
            skipped_items(skipped),
        )
    else:
        notify("ok", t("added", name=name, count=count))
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
        notify("error", t("delete_failed", name=file_name, error=error))
        return
    refresh_after_change()
    notify("ok", t("deleted", name=file_name))


def added_time(uploaded_at):
    return f"{uploaded_at[:16].replace('T', ' ')} UTC" if uploaded_at else ""


def row_details(document):
    return f"{t('col_chunks')} {document['chunks']} · {added_time(document['uploaded_at'])}"


def render_document_row(document):
    doc_id, name = document["doc_id"], document["file_name"]
    with st.container(key=f"row-{doc_id}"):
        file, status, chunks, added, action = st.columns(TABLE_COLUMNS, vertical_alignment="center")
        file.markdown(
            '<div class="pd-file"><span class="pd-pdf">PDF</span><div>'
            f'<div class="pd-file-name">{escape(name)}</div>'
            f'<div class="pd-file-id pd-muted">{escape(doc_id)}</div>'
            f'<div class="pd-mobile-only pd-muted">{escape(row_details(document))}</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        skipped = document["skipped_figures"]
        chip = (
            f'<span class="pd-chip warn">{escape(t("status_skipped", count=skipped))}</span>'
            if skipped
            else f'<span class="pd-chip ok">{escape(t("status_ok"))}</span>'
        )
        status.markdown(chip, unsafe_allow_html=True)
        chunks.markdown(
            f'<span class="pd-num pd-desktop-only">{document["chunks"]}</span>',
            unsafe_allow_html=True,
        )
        added.markdown(
            '<span class="pd-num pd-muted pd-desktop-only">'
            f"{escape(added_time(document['uploaded_at']))}</span>",
            unsafe_allow_html=True,
        )
        action.button(t("delete"), key=f"delete-{doc_id}", on_click=request_delete, args=(doc_id,))
    if st.session_state.get("pending_delete") == doc_id:
        with st.container(key=f"confirm-{doc_id}"):
            message, cancel, confirm = st.columns([5, 1, 1], vertical_alignment="center")
            message.markdown(escape(t("confirm_delete", name=name)))
            cancel.button(t("cancel"), key=f"cancel-{doc_id}", on_click=cancel_delete)
            confirm.button(
                t("delete"),
                key=f"really-delete-{doc_id}",
                type="primary",
                on_click=delete_document,
                args=(doc_id, name),
            )


def render_documents(documents):
    st.markdown(
        f'<div class="pd-h2">{escape(t("docs_title"))}</div>'
        f'<p class="pd-muted">{escape(t("docs_intro"))}</p>',
        unsafe_allow_html=True,
    )
    for notice in st.session_state.pop("notices", []):
        st.markdown(notice_html(notice), unsafe_allow_html=True)

    with st.container(border=True, key="upload"):
        picker, add = st.columns([4, 1.3], vertical_alignment="center")
        upload = picker.file_uploader(
            t("upload_label"),
            type=["pdf"],
            key=f"upload-{st.session_state.get('upload_round', 0)}",
            label_visibility="collapsed",
        )
        picker.caption(t("upload_hint"))
        clicked = add.button(
            t("add_button"), type="primary", disabled=upload is None, width="stretch"
        )
    progress_slot = st.empty()
    if clicked:
        add_upload(upload, progress_slot)

    if documents is None:
        st.caption(t("list_unavailable"))
        return
    if not documents:
        st.info(t("no_docs"))
        return

    with_skipped = [document for document in documents if document["skipped_figures"]]
    # Labels are computed once per run; the widget value travels as its label.
    filters = {
        "all": t("filter_all", count=len(documents)),
        "skipped": t("filter_skipped", count=len(with_skipped)),
    }
    shown = st.segmented_control(
        t("docs_title"),
        options=list(filters),
        format_func=filters.get,
        key="doc_filter",
        required=True,
        label_visibility="collapsed",
    )
    rows = with_skipped if shown == "skipped" else documents
    with st.container(border=True, key="doc_table"):
        headings = ("col_file", "col_status", "col_chunks", "col_added", None)
        with st.container(key="doc_head"):
            for column, key in zip(st.columns(TABLE_COLUMNS), headings):
                if key:
                    column.markdown(
                        f'<div class="pd-th pd-muted">{escape(t(key))}</div>',
                        unsafe_allow_html=True,
                    )
        if not rows:
            st.caption(t("no_skipped"))
        for document in rows:
            render_document_row(document)
    st.caption(t("docs_foot"))


def render_header(documents):
    brand, middle, right = st.columns([1, 1.6, 1], vertical_alignment="center")
    brand.markdown(
        '<div class="pd-brand"><span class="pd-mark"></span>PolicyDesk</div>',
        unsafe_allow_html=True,
    )
    with middle.container(horizontal=True, horizontal_alignment="center"):
        views = {"ask": t("nav_ask"), "docs": t("nav_docs", count=len(documents or []))}
        st.segmented_control(
            t("nav_ask"),
            options=list(views),
            format_func=views.get,
            key="view",
            required=True,
            label_visibility="collapsed",
        )
    with right.container(horizontal=True, horizontal_alignment="right"):
        st.segmented_control(
            t("language"),
            options=list(LANGUAGES),
            format_func=LANGUAGES.get,
            key="language",
            required=True,
            label_visibility="collapsed",
        )


def main():
    st.set_page_config(
        page_title="PolicyDesk | Policy assistant",
        page_icon="📘",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.session_state.setdefault("language", DEFAULT_LANGUAGE)
    st.session_state.setdefault("view", "ask")
    try:
        documents = list_documents()
    except Exception:
        LOGGER.exception("Could not read the document list")
        documents = None

    width = "1040px" if st.session_state["view"] == "docs" else "820px"
    st.markdown(STYLE.replace("MAX_WIDTH", width), unsafe_allow_html=True)
    render_header(documents)
    if documents is None:
        st.error(t("db_error"))
    if st.session_state["view"] == "docs":
        render_documents(documents)
    else:
        render_ask(documents)


if __name__ == "__main__":
    main()
