# PolicyDesk

PolicyDesk is a small internal policy assistant. A company adds its own policy PDFs,
then employees ask questions in Chinese or English and see the retrieved evidence with
file names and page numbers. It has a local web demo and persistent CLI conversations.

## Implemented architecture

```text
Policy PDF (web upload or `python main.py add`)
  -> MinerU content list: text, HTML tables, figures, page numbers
  -> figures described by Qwen-VL (failed figures are skipped and reported)
  -> per-page text / table / figure blocks -> SentenceSplitter (512 tokens, 50 overlap)
  -> appended to MongoDB nodes + Redis index structure + Chroma embeddings

Question (Chinese or English)
  -> one guarded Qwen query rewrite in the question's language
  -> vector search + BM25 (Chinese characters and English words)
  -> reciprocal rank fusion -> BGE reranking -> Qwen answer in the question's language
```

- **Document management:** PDFs are added and deleted one at a time; the index is
  updated in place rather than rebuilt. Each PDF gets a `doc_id` derived from its
  content (`pdf-<sha256 prefix>`), so the same file cannot be added twice. Deleting a
  PDF removes its vectors, node references, and nodes from all three stores.
- **Parsing:** MinerU 4.x runs in its own environment so its heavy dependencies stay
  out of this one. PolicyDesk calls `mineru_content_list.py` with that environment's
  Python; the script uses MinerU's Python API to write a content list (MinerU 4.x no
  longer has a command that does this). Page headers, footers, and page numbers
  are dropped. Text is grouped per page; each table and figure becomes its own block.
  Tables keep MinerU's HTML. If Qwen-VL cannot describe a figure, that figure is
  skipped, the reason is shown, and the rest of the PDF is still added.
- **Node metadata:** `file_name`, `page`, `content_type` (`text`, `table`, `figure`),
  `policy_id` (the `doc_id`), `uploaded_at`, and `skipped_figures`. Page numbers are
  kept out of embeddings; upload bookkeeping is kept out of embeddings and prompts.
- **Query preparation:** one Qwen rewrite prepares a search query in the question's
  language while the original question remains available for reranking and answer
  generation. Invalid, multiline, oversized, or numerically changed rewrites fall back
  to the original; provider failures also fall back.
- **Hybrid retrieval:** vector and BM25 searches each return up to eight candidates.
  BM25 treats each Chinese character and each English word as a keyword. Reciprocal
  rank fusion deduplicates evidence and keeps up to ten. BM25 is rebuilt from the
  database's indexed nodes.
- **Post-retrieval processing:** local `BAAI/bge-reranker-large` keeps up to four
  excerpts. Displayed scores are reranker relevance scores, not probabilities.
- **Answers and sessions:** Qwen Plus streams answers in the question's language.
  CLI chat uses question condensation and a Redis-backed 3,000-token memory window per
  knowledge base and session. Empty model output raises an actionable error.
- **Web demo:** sidebar upload, a document list with confirmed deletion, independent
  questions, streamed answers, and expandable excerpts with file, page, and type.
  Models load once per process; the search engine is rebuilt automatically after a
  PDF is added or deleted.

| Module | Responsibility |
| --- | --- |
| `pdf_ingestion.py` | MinerU parsing, figure descriptions, adding, listing, and deleting PDFs |
| `mineru_content_list.py` | Runs inside MinerU's environment and writes the content list |
| `retrieval.py` | Query rewriting, BM25, rank fusion, and optional metadata scopes |
| `services.py` | Splitting, query engine assembly, and chat memory |
| `base_llm.py` | Qwen, embedding, and reranker factories |
| `storage.py` | MongoDB, Redis, and Chroma connections |
| `main.py` | CLI: add, delete, list, ask, and persistent chat |
| `app.py` | Streamlit interface and resource caching |
| `prompts.py` | Answer and rewrite prompts, plus reusable template variants |
| `ingestion.py`, `data/` | Sample text policies used only by the offline tests |

The knowledge base contains only PDFs you add. The sample policies in `data/` are test
fixtures and are never indexed by the app or CLI.

## Installation

Requirements: Python 3.11, Docker with Compose, a DashScope key authorized for
`qwen-plus` and the vision model (`qwen3-vl-plus` by default), and MinerU. Internet
access is needed for dependency installation, Qwen calls, and initial HuggingFace model
downloads. Embedding and reranker weights run locally and can use substantial memory.

Run these commands from the PolicyDesk directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For an existing uv-managed environment without pip, use
`uv pip install --python .venv/bin/python -r requirements.txt` instead.
Development dependencies are in `requirements-dev.txt`.

Install MinerU 4.x in **its own** environment so its dependencies do not conflict with
this project's, for example:

```bash
conda create -n mineru python=3.11 -y
conda run -n mineru python -m pip install -U "mineru==4.2.0"
conda run -n mineru which mineru
```

Set `MINERU_COMMAND` to the full path printed by the last command, for example
`/opt/anaconda3/envs/mineru/bin/mineru`. PolicyDesk runs `mineru_content_list.py` with
the `python` in the same folder (override it with `MINERU_PYTHON`). The first parse
downloads MinerU's models. Alternatively, parse a
PDF elsewhere (for example with MinerU's online service) and pass its
`*_content_list.json` to `python main.py add --content-list` together with its
`images/` folder.

For a new installation, copy `.env.example` to `.env`. If `.env` already exists,
edit it and preserve existing settings. Obtain a key through the
[official Model Studio API key guide](https://help.aliyun.com/zh/model-studio/get-api-key).

```dotenv
DASHSCOPE_API_KEY=your_dashscope_api_key
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
RERANKER_MODEL=BAAI/bge-reranker-large
KB_NAME=policydesk_uploads_v1
MONGO_DB=policydesk
MINERU_COMMAND=mineru
VL_MODEL=qwen3-vl-plus
```

| Variable | Meaning |
| --- | --- |
| `DASHSCOPE_API_KEY` | Required for Qwen answers, query rewriting, and figure descriptions |
| `EMBEDDING_MODEL` | HuggingFace model ID or local directory; default shown above |
| `RERANKER_MODEL` | HuggingFace reranker ID or local directory; default shown above |
| `KB_NAME` | Namespace for one company's nodes, vectors, index structure, and chat keys |
| `MONGO_DB` | MongoDB database name; default `policydesk` |
| `MINERU_COMMAND` | Full path of the `mineru` executable in MinerU's environment |
| `MINERU_PYTHON` | Optional Python of MinerU's environment; default: beside `MINERU_COMMAND` |
| `MINERU_TIER` | Optional MinerU parse tier: `flash`, `basic`, `standard` (default), `advanced` |
| `VL_MODEL` | DashScope vision model that describes figures |
| `COMPOSE_PROJECT_NAME` | Optional existing Docker project name; retain it when reusing volumes |

The code loads `.env` beside the Python modules. Shell variables take precedence.
Keep real keys out of source control. The DashScope adapter uses its default China
(Beijing) endpoint, so the key must be valid for that region. Figure images are uploaded
to DashScope for description. Switching embedding models requires a new `KB_NAME`.

**Upgrading from the sample-policy version:** an existing namespace such as
`policydesk_metadata_v1` still contains the eight sample policies. Set a new, unused
`KB_NAME` (for example `policydesk_uploads_v1`) so that the knowledge base starts empty.
Old namespaces are retained and not deleted.

Start the services:

```bash
docker compose up -d --wait
docker compose ps
```

MongoDB, Redis, and Chroma bind to localhost ports 27028, 6389, and 8018.
Named volumes persist data.

## Run

```bash
python -m streamlit run app.py
python main.py add /path/to/travel_policy.pdf
python main.py add /path/to/leave_policy.pdf --content-list /path/to/leave_policy_content_list.json
python main.py list
python main.py ask "出差住宿标准是多少？"
python main.py ask "Who approves four working days of leave?"
python main.py chat --session employee-travel
python main.py delete pdf-0123456789abcdef
```

- In [the local demo](http://127.0.0.1:8501), add a PDF in the sidebar, wait for
  parsing to finish, then ask questions. Delete asks for confirmation. Skipped figures
  are listed after the upload and counted in the document list.
- `list` shows each document ID, file name, chunk count, upload time, and skipped
  figures. Use the ID with `delete`. Deleted PDFs can be added again.
- Adding the same file twice is rejected before any parsing or model calls.
- Questions are rejected until at least one PDF has been added. Enter `/exit` to leave
  CLI chat; reuse the same session to restore history.
- The web demo rebuilds its search engine after its own uploads and deletions. Restart
  Streamlit after adding or deleting PDFs from the CLI so keyword search sees them.

## Testing

Install development dependencies, then run the offline suite and code checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Offline tests use model and storage doubles and need no API key, MinerU, model
downloads, or running databases. They cover PDF parsing with skipped figures; adding,
listing, deleting, and re-adding PDFs while Chroma, the docstore, and the index
structure stay consistent; duplicate and empty PDFs; Chinese and English keyword
search; rewrite fallback; original-question preservation; conversation memory; and
Streamlit upload, deletion, streaming, and error behavior.

With the three database services running:

```bash
POLICYDESK_INTEGRATION=1 python -m pytest -k database -q
```

This builds and reloads an isolated `policydesk_test_*` namespace from the sample text
policies, verifies metadata filtering through Chroma and BM25, and restores Redis chat
history. It uses test-only models and does not call Qwen or MinerU.

If Qwen returns no answer, check the key, region, and account quota. In particular,
`AllocationQuota.FreeTierOnly` means that the account has exhausted its free quota
while paid usage is disabled.

For a real-model smoke test, add a PDF that contains a flowchart and a table, then ask
questions answered only by the text, only by the table, and only by the figure, in both
Chinese and English. Check that the excerpts show the correct file and page.

## Operations and limitations

```bash
docker compose stop
docker compose start
docker compose logs --tail 30
```

Keep Docker volumes to retain data; do not use volume-removal commands for routine
restarts. Stop Streamlit with Ctrl+C.

This remains a simplified local application without user authentication or access
control. One `KB_NAME` holds one company's documents; serving several companies needs
one deployment (or process) per `KB_NAME`. Adding and deleting are serialized within
one process; do not change the same knowledge base from the CLI and the web demo at
the same time. Updates across the three databases are not transactional. Chinese
keyword search matches single characters, which is coarser than word segmentation.
Policy excerpts are supporting evidence rather than verified sentence-level citations;
unresolved policy questions should go to the relevant policy owner.

## Future work

Possible follow-up improvements include a representative bilingual evaluation set,
Markdown conversion and splitting for large tables, section headings in metadata,
Chinese word segmentation for keyword search, background ingestion jobs, several
companies per deployment with authentication, and explicit version replacement.
