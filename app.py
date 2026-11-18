"""Web demo: ask about company documents, and add or delete them.

The policy domain answers from uploaded policy PDFs. The on-call domain (KB_DOMAIN=oncall)
adds runbooks, postmortems, the service catalog, a retrieval-mode switch, and a side-by-side
comparison of hybrid retrieval with hybrid + graph retrieval.
"""

import html
import json
import logging
import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import streamlit as st
from llama_index.core import load_index_from_storage
from llama_index.core.schema import MetadataMode

import graph_retrieval
import kb
import services
import storage
from base_llm import create_models, create_reranker
from metrics import TOKEN_METER, AnswerTimer, format_seconds, log_answer
from ui_text import DEFAULT_LANGUAGE, LANGUAGES, translate

LOGGER = logging.getLogger(__name__)
KIND_KEYS = {"text": "kind_text", "table": "kind_table", "figure": "kind_figure"}
UPLOAD_STEPS = ("upload", "parse", "describe", "index")
MARKDOWN_STEPS = ("upload", "parse", "extract", "index")
TABLE_COLUMNS = [4, 2.2, 1, 2, 1.2]
FILE_BADGES = {"pdf": "PDF", "runbook": "MD", "postmortem": "MD", "service_catalog": "YAML"}
STRUCTURED_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```\s*$", re.DOTALL)

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
.pd-graph {border-left: 3px solid var(--pd-accent); background: var(--pd-surface);
  border-radius: 0 10px 10px 0; padding: .5rem .8rem; margin: .35rem 0; font-size: .84rem;
  font-family: ui-monospace, Menlo, monospace; white-space: pre-wrap; word-break: break-word;}
.pd-timing {font-size: .76rem; margin: .3rem 0 .6rem;}
.pd-mode {font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  color: var(--pd-accent); margin-top: .6rem;}
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
    language = st.session_state.get("language", DEFAULT_LANGUAGE)
    return translate(language, key, domain=storage.KB_DOMAIN, **values)


def escape(value):
    return html.escape(str(value))


@st.cache_resource(show_spinner=False)
def load_models():
    """Load Qwen, the embedding model, and the reranker once per process."""
    llm, embed_model = create_models()
    services.configure_settings(llm, embed_model)
    TOKEN_METER.attach(llm)
    return llm, create_reranker()


@st.cache_resource(show_spinner=False)
def load_graph(knowledge_base):
    """(graph_store, catalog) for the on-call domain; cleared after a catalog import."""
    return kb.connect_graph()


def graph_resources():
    """Return (graph_store, catalog, problem). problem is None when the graph is usable."""
    if not kb.is_oncall():
        return None, None, None
    try:
        graph_store, catalog = load_graph(storage.KB_NAME)
    except Exception as error:
        LOGGER.exception("Graph store unavailable")
        return None, None, str(error)
    return graph_store, catalog, None


@st.cache_resource(show_spinner=False)
def load_extraction(knowledge_base):
    llm, _ = load_models()
    return kb.build_extraction(llm)


@st.cache_resource(show_spinner=False)
def load_query_engine(knowledge_base, mode="hybrid"):
    """Cache one engine per mode until a document changes; BM25 uses this snapshot."""
    context, _ = storage.connect_storage(services)
    if not context.index_store.get_index_struct(storage.INDEX_ID):
        raise ValueError(f"Add a document to {knowledge_base} before asking questions.")
    llm, reranker = load_models()
    graph_store, catalog, _ = graph_resources()
    graph_search = kb.graph_search_for(graph_store, catalog)
    resolved = graph_retrieval.resolve_mode(mode, graph_search is not None)
    index = load_index_from_storage(context, index_id=storage.INDEX_ID)
    return kb.build_query_engine(index, llm, reranker, resolved, graph_search), resolved


@st.cache_resource(show_spinner=False)
def connect_docstore(knowledge_base):
    """Reuse one MongoDB client for the document list."""
    return storage.connect_docstore()


def list_documents():
    return kb.list_documents(connect_docstore(storage.KB_NAME))


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


def split_answer(text):
    """Separate the trailing ```json block (for evaluation) from the visible answer."""
    match = STRUCTURED_BLOCK.search(text or "")
    if not match:
        return text, None
    try:
        structured = json.loads(match.group(1))
    except json.JSONDecodeError:
        return text, None
    return text[: match.start()].rstrip(), structured


def source_details(response):
    """Snapshot retrieved text and scores for display across Streamlit reruns."""
    return [
        {
            "name": node.metadata.get("file_name", "Unknown source"),
            "page": node.metadata.get("page"),
            "section": node.metadata.get("section"),
            "kind": node.metadata.get("content_type", "text"),
            "score": node.score,
            "text": plain_text(node.get_content(metadata_mode=MetadataMode.NONE)),
        }
        for node in response.source_nodes
    ]


def source_location(source):
    parts = [escape(source["name"])]
    if source.get("page") is not None:
        parts.append(escape(t("page", page=source["page"])))
    elif source.get("section"):
        parts.append(escape(snippet(source["section"], 40)))
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


def render_source_cards(sources, columns=4):
    documents = [source for source in sources if source["kind"] != "graph_path"]
    st.markdown(
        f'<div class="pd-label pd-muted">{escape(t("sources_label", count=len(documents)))}</div>',
        unsafe_allow_html=True,
    )
    if not documents:
        st.info(t("no_sources"))
    else:
        cards = "".join(
            '<div class="pd-card">'
            f'<div class="pd-card-text">{escape(snippet(source["text"]))}</div>'
            '<div class="pd-card-meta pd-muted">'
            f'<span class="pd-kind">{escape(kind_label(source["kind"]))}'
            f"</span>{source_location(source)}</div></div>"
            for source in documents
        )
        grid = f"grid-template-columns: repeat({columns}, minmax(0, 1fr));"
        st.markdown(f'<div class="pd-cards" style="{grid}">{cards}</div>', unsafe_allow_html=True)
    render_graph_paths(sources)


def render_graph_paths(sources):
    paths = [source for source in sources if source["kind"] == "graph_path"]
    if not paths:
        return
    st.markdown(
        f'<div class="pd-label pd-muted">{escape(t("graph_paths_label", count=len(paths)))}</div>'
        + "".join(f'<div class="pd-graph">{escape(source["text"])}</div>' for source in paths),
        unsafe_allow_html=True,
    )


def render_excerpts(sources):
    documents = [source for source in sources if source["kind"] != "graph_path"]
    if not documents:
        return
    with st.expander(t("excerpts")):
        for number, source in enumerate(documents, start=1):
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


def render_timing(run):
    timing, tokens = run["timing"], run["tokens"]
    text = t(
        "timing",
        retrieval=format_seconds(timing["retrieval_s"]),
        first=format_seconds(timing["first_token_s"]),
        generation=format_seconds(timing["generation_s"]),
        total=format_seconds(timing["total_s"]),
        calls=tokens["llm_calls"],
        tokens=tokens["prompt_tokens"] + tokens["completion_tokens"],
    )
    st.markdown(f'<div class="pd-timing pd-muted">{escape(text)}</div>', unsafe_allow_html=True)


def render_structured(structured):
    if structured is not None:
        with st.expander(t("structured_label")):
            st.json(structured)


def answer_one(question, mode, columns):
    """Run one mode, streaming into the current container. Returns the saved run."""
    phases = st.empty()
    render_phases(phases, "retrieve")
    TOKEN_METER.reset()
    timer = AnswerTimer()
    engine, resolved = load_query_engine(storage.KB_NAME, mode)
    if kb.is_oncall():
        st.markdown(
            f'<div class="pd-mode">{escape(t("mode_" + resolved))}</div>', unsafe_allow_html=True
        )
    response = engine.query(question)
    timer.mark_retrieved()
    sources = source_details(response)
    render_source_cards(sources, columns)
    render_phases(phases, "write")
    st.markdown(
        f'<div class="pd-label pd-muted">{escape(t("answer_label"))}</div>',
        unsafe_allow_html=True,
    )
    with st.container(key=f"answer-{resolved}"):
        slot = st.empty()
        text = slot.write_stream(timer.stream(response.response_gen))
        body, structured = split_answer(text)
        if structured is not None:
            slot.markdown(body)
    render_phases(phases, "done")
    run = {
        "mode": resolved,
        "answer": body,
        "structured": structured,
        "sources": sources,
        "timing": timer.summary(),
        "tokens": TOKEN_METER.snapshot(),
    }
    log_answer(resolved, run["timing"], run["tokens"])
    render_timing(run)
    render_structured(structured)
    render_excerpts(sources)
    return run


def selected_modes():
    if not kb.is_oncall():
        return ["hybrid"]
    if st.session_state.get("compare"):
        return ["hybrid", "hybrid+graph"]
    return [st.session_state.get("mode", "hybrid+graph")]


def answer(question):
    st.session_state.pop("result", None)
    area = st.empty()
    modes = selected_modes()
    try:
        with area.container():
            columns = st.columns(len(modes)) if len(modes) > 1 else [st.container()]
            runs = []
            for mode, column in zip(modes, columns):
                with column:
                    runs.append(answer_one(question, mode, 2 if len(modes) > 1 else 4))
            st.session_state["result"] = {"question": question, "runs": runs}
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
    runs = result["runs"]
    columns = st.columns(len(runs)) if len(runs) > 1 else [st.container()]
    for run, column in zip(runs, columns):
        with column:
            if kb.is_oncall():
                st.markdown(
                    f'<div class="pd-mode">{escape(t("mode_" + run["mode"]))}</div>',
                    unsafe_allow_html=True,
                )
            render_source_cards(run["sources"], 2 if len(runs) > 1 else 4)
            st.markdown(
                f'<div class="pd-label pd-muted">{escape(t("answer_label"))}</div>',
                unsafe_allow_html=True,
            )
            with st.container(key=f"answer-{run['mode']}"):
                st.markdown(run["answer"])
            render_timing(run)
            render_structured(run["structured"])
            render_excerpts(run["sources"])


def ask_example(question):
    st.session_state["question"] = question
    st.session_state["auto_ask"] = True


def show_view(view):
    st.session_state["view"] = view


def render_mode_controls():
    """On-call only: pick a retrieval mode, or compare hybrid with hybrid + graph."""
    _, catalog, problem = graph_resources()
    if problem or catalog is None:
        st.caption(t("graph_unavailable", reason=problem or t("catalog_missing")))
    modes = {mode: t("mode_" + mode) for mode in graph_retrieval.MODES}
    left, right = st.columns([3, 2], vertical_alignment="center")
    with left:
        st.segmented_control(
            t("mode_label"),
            options=list(modes),
            format_func=modes.get,
            key="mode",
            required=True,
            disabled=bool(st.session_state.get("compare")),
            label_visibility="collapsed",
        )
    right.toggle(t("compare"), key="compare")


def render_ask(documents):
    st.markdown(f'<div class="pd-h1">{escape(t("ask_title"))}</div>', unsafe_allow_html=True)
    if documents == []:
        with st.container(border=True):
            st.markdown(t("empty_ask"))
            st.button(t("go_docs"), type="primary", on_click=show_view, args=("docs",))
    if kb.is_oncall():
        render_mode_controls()

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


def progress_html(name, steps, step):
    current = steps.index(step)
    spans = "".join(
        f'<span class="pd-step {"done" if index < current else "now" if index == current else ""}">'
        f"{escape(t('step_' + key))}</span>"
        for index, key in enumerate(steps)
    )
    return (
        '<div class="pd-progress"><div class="pd-progress-top">'
        f"<span>{escape(t('adding', name=name))}</span>"
        f'<span class="pd-muted">{escape(t("adding_hint"))}</span></div>'
        f'<div class="pd-steps pd-muted">{spans}</div><div class="pd-bar"><i></i></div></div>'
    )


def refresh_after_change():
    """Rebuild the engines on the next question so every search branch sees the change."""
    load_query_engine.clear()
    st.session_state.pop("result", None)


def ingest_upload(name, data, progress=None, doc_type=None):
    """Save the upload under its own name so sources show it, then append it.

    Returns (doc_id, chunk count, report) like kb.add_document.
    """
    llm, _ = load_models()  # Configures the embedding model used for indexing.
    context, collection = storage.connect_storage(services, create=True)
    graph_store, catalog, problem = graph_resources()
    is_markdown = Path(name).suffix.lower() in kb.MARKDOWN_SUFFIXES
    if kb.is_oncall() and problem:
        raise ValueError(t("needs_graph", reason=problem))
    extraction = load_extraction(storage.KB_NAME) if is_markdown and graph_store else None
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / Path(name).name
        path.write_bytes(data)
        return kb.add_document(
            path,
            context,
            collection,
            work_dir=Path(folder) / "mineru",
            doc_type=doc_type,
            graph_store=graph_store,
            catalog=catalog,
            extraction=extraction,
            progress=progress,
        )


def skipped_items(skipped):
    items = []
    for item in skipped:
        caption = f" · {item['caption']}" if item["caption"] else ""
        reason = item["reason"][:200]
        items.append(t("skipped_item", page=item["page"], caption=caption, reason=reason))
    return items


def add_upload(upload, slot, doc_type=None):
    name = upload.name
    is_markdown = Path(name).suffix.lower() in kb.MARKDOWN_SUFFIXES
    steps = MARKDOWN_STEPS if is_markdown else UPLOAD_STEPS

    def show(step):
        if step in steps:
            slot.markdown(progress_html(name, steps, step), unsafe_allow_html=True)

    show("upload")
    try:
        _, count, report = ingest_upload(name, upload.getvalue(), progress=show, doc_type=doc_type)
    except Exception as error:
        LOGGER.exception("Document ingestion failed")
        failure = {"kind": "error", "title": t("add_failed", name=name, error=error)}
        slot.markdown(notice_html({**failure, "body": "", "items": []}), unsafe_allow_html=True)
        return
    refresh_after_change()
    skipped = report.get("skipped", [])
    if skipped:
        notify(
            "warn",
            t("added", name=name, count=count),
            t("skipped", count=len(skipped)),
            skipped_items(skipped),
        )
    elif "relations" in report:
        dropped = ", ".join(report["dropped"][:10]) or "–"
        notify(
            "ok",
            t("added", name=name, count=count),
            t(
                "graph_report",
                relations=report["relations"],
                empty=report["empty_chunks"],
                dropped=dropped,
            ),
        )
    else:
        notify("ok", t("added", name=name, count=count))
    # A new uploader key clears the selected file after a successful add.
    st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1
    st.rerun()


def import_catalog_upload(catalog_file, alerts_file, slot):
    try:
        load_models()  # Configures the embedding model for the catalog documents.
        context, collection = storage.connect_storage(services, create=True)
        graph_store, _, problem = graph_resources()
        if problem:
            raise ValueError(t("needs_graph", reason=problem))
        alerts = alerts_file.getvalue().decode("utf-8") if alerts_file else ""
        if catalog_file is not None:
            catalog_text = catalog_file.getvalue().decode("utf-8")
        else:
            # Only new alerts: keep the service catalog that is already in Neo4j.
            source = kb.service_catalog.catalog_source(graph_store)
            if source is None:
                raise ValueError(t("catalog_needs_file"))
            catalog_text = source["catalog"]
        catalog, _ = kb.import_catalog(catalog_text, alerts, context, collection, graph_store)
    except Exception as error:
        LOGGER.exception("Catalog import failed")
        failure = {"kind": "error", "title": t("catalog_failed", error=error)}
        slot.markdown(notice_html({**failure, "body": "", "items": []}), unsafe_allow_html=True)
        return
    load_graph.clear()
    refresh_after_change()
    notify(
        "ok",
        t(
            "catalog_imported",
            services=len(catalog.components),
            teams=len(catalog.teams),
            alerts=len(catalog.alerts),
        ),
    )
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
        graph_store, _, problem = graph_resources()
        if kb.is_oncall() and problem:
            raise ValueError(t("needs_graph", reason=problem))
        kb.delete_document(context, doc_id, graph_store)
    except Exception as error:
        LOGGER.exception("Document deletion failed")
        notify("error", t("delete_failed", name=file_name, error=error))
        return
    refresh_after_change()
    notify("ok", t("deleted", name=file_name))


def added_time(uploaded_at):
    return f"{uploaded_at[:16].replace('T', ' ')} UTC" if uploaded_at else ""


def row_details(document):
    return f"{t('col_chunks')} {document['chunks']} · {added_time(document['uploaded_at'])}"


def status_chip(document):
    if document.get("services"):
        return "ok", t("status_services", count=document["services"])
    if document["skipped_figures"]:
        return "warn", t("status_skipped", count=document["skipped_figures"])
    if document.get("graph_relations"):
        return "ok", t("status_relations", count=document["graph_relations"])
    return "ok", t("status_ok")


def render_document_row(document):
    doc_id, name = document["doc_id"], document["file_name"]
    doc_type = document.get("doc_type", "pdf")
    with st.container(key=f"row-{doc_id}"):
        file, status, chunks, added, action = st.columns(TABLE_COLUMNS, vertical_alignment="center")
        file.markdown(
            f'<div class="pd-file"><span class="pd-pdf">{FILE_BADGES.get(doc_type, "DOC")}</span>'
            f'<div><div class="pd-file-name">{escape(name)}</div>'
            f'<div class="pd-file-id pd-muted">{escape(t("type_" + doc_type))} · '
            f"{escape(doc_id)}</div>"
            f'<div class="pd-mobile-only pd-muted">{escape(row_details(document))}</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        kind, label = status_chip(document)
        status.markdown(
            f'<span class="pd-chip {kind}">{escape(label)}</span>', unsafe_allow_html=True
        )
        chunks.markdown(
            f'<span class="pd-num pd-desktop-only">{document["chunks"]}</span>',
            unsafe_allow_html=True,
        )
        added.markdown(
            '<span class="pd-num pd-muted pd-desktop-only">'
            f"{escape(added_time(document['uploaded_at']))}</span>",
            unsafe_allow_html=True,
        )
        if doc_id != kb.CATALOG_ROW_ID:
            action.button(
                t("delete"), key=f"delete-{doc_id}", on_click=request_delete, args=(doc_id,)
            )
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


def render_catalog_import():
    """On-call only: upload service_catalog.yaml (and optional alerts.yaml)."""
    _, catalog, problem = graph_resources()
    with st.container(border=True, key="catalog"):
        st.markdown(f"**{escape(t('catalog_title'))}**")
        if catalog is not None:
            st.caption(
                t("catalog_status", services=len(catalog.components), alerts=len(catalog.alerts))
            )
        else:
            st.caption(problem or t("catalog_missing"))
        round_id = st.session_state.get("upload_round", 0)
        left, right, button = st.columns([2, 2, 1.3], vertical_alignment="bottom")
        catalog_file = left.file_uploader(
            t("catalog_title"), type=["yaml", "yml"], key=f"catalog-{round_id}"
        )
        alerts_file = right.file_uploader(
            t("catalog_alerts"), type=["yaml", "yml"], key=f"alerts-{round_id}"
        )
        # Alerts alone update the imported catalog's alerts; a first import needs the catalog.
        ready = catalog_file is not None or (alerts_file is not None and catalog is not None)
        clicked = button.button(
            t("catalog_button"), disabled=not ready, width="stretch", key="import"
        )
        if alerts_file is not None and catalog_file is None:
            st.caption(t("alerts_only") if catalog is not None else t("catalog_needs_file"))
    slot = st.empty()
    if clicked:
        import_catalog_upload(catalog_file, alerts_file, slot)


def render_documents(documents):
    st.markdown(
        f'<div class="pd-h2">{escape(t("docs_title"))}</div>'
        f'<p class="pd-muted">{escape(t("docs_intro"))}</p>',
        unsafe_allow_html=True,
    )
    for notice in st.session_state.pop("notices", []):
        st.markdown(notice_html(notice), unsafe_allow_html=True)
    if kb.is_oncall():
        render_catalog_import()

    file_types = ["pdf", "md", "markdown"] if kb.is_oncall() else ["pdf"]
    with st.container(border=True, key="upload"):
        picker, add = st.columns([4, 1.3], vertical_alignment="center")
        upload = picker.file_uploader(
            t("upload_label"),
            type=file_types,
            key=f"upload-{st.session_state.get('upload_round', 0)}",
            label_visibility="collapsed",
        )
        picker.caption(t("upload_hint"))
        doc_type = None
        if kb.is_oncall():
            md_types = {"auto": t("md_auto"), "runbook": t("md_runbook")}
            md_types["postmortem"] = t("md_postmortem")
            choice = picker.segmented_control(
                t("md_type"),
                options=list(md_types),
                format_func=md_types.get,
                key="md_type",
                required=True,
            )
            doc_type = None if choice == "auto" else choice
        clicked = add.button(
            t("add_button"), type="primary", disabled=upload is None, width="stretch"
        )
    progress_slot = st.empty()
    if clicked:
        add_upload(upload, progress_slot, doc_type)

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
    st.session_state.setdefault("mode", "hybrid+graph")
    try:
        documents = list_documents()
    except Exception:
        LOGGER.exception("Could not read the document list")
        documents = None

    wide = st.session_state["view"] == "docs" or st.session_state.get("compare")
    width = "1180px" if st.session_state.get("compare") else "1040px" if wide else "820px"
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
