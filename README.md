# PolicyDesk

PolicyDesk is a small internal policy assistant. A company adds its own policy PDFs,
then employees ask questions in Chinese or English and see the retrieved evidence with
file names and page numbers. It has a local web demo and persistent CLI conversations.

Version 2 adds an **on-call domain** (`KB_DOMAIN=oncall`): a service catalog, runbooks
and postmortems in Markdown, and a Neo4j knowledge graph. On-call questions can be
answered with the existing hybrid retrieval, with graph retrieval, or with both, so the
graph's contribution can be measured against the same baseline. The policy domain is
unchanged and remains the default.

## Implemented architecture

### Policy domain (default)

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
- **Web demo:** two views, Ask and Documents, with a Chinese/English interface switch
  (answers always follow the question's language) and Streamlit's light or dark theme.
  Ask shows search, ranking, and writing phases, source cards with file, page, and
  type before the streamed answer, expandable excerpts, and example questions.
  Documents has an upload area with step-by-step progress, notices for skipped
  figures, a filter for documents with skipped figures, and a table with status,
  chunk count, time added, and confirmed deletion. Models load once per process; the
  search engine is rebuilt automatically after a PDF is added or deleted. The design
  is `design/mockups/d_answer_engine_with_table.html`.

### On-call domain (`KB_DOMAIN=oncall`)

```text
service_catalog.yaml (+ alerts.yaml)          runbooks/*.md, postmortems/*.md
  -> validated, imported to Neo4j directly      -> front matter + heading-aware chunks
     (no LLM): services, datastores, teams,     -> MongoDB + Redis + Chroma (same chunk IDs)
     DEPENDS_ON / OWNED_BY / ESCALATES_TO /     -> postmortems only: schema-constrained
     FIRES_ON / HANDLED_BY                         Qwen extraction (strict) -> aligned to
  -> also rendered as one text document per        catalog names and the symptom vocabulary
     component for the hybrid baseline             -> Neo4j, with doc_ids/chunk_ids on facts

Question -> catalog alias linking on the ORIGINAL question -> intent (impact, root cause,
owner, similar incidents, alert) -> fixed Cypher templates (the LLM never writes Cypher)
  hybrid:        the policy-domain chain, unchanged (vector + BM25 -> RRF -> reranker)
  graph:         passages behind the graph facts -> reranker -> + graph facts
  hybrid+graph:  vector + BM25 + graph passages -> RRF -> reranker -> + graph facts
  -> Qwen answer in the question's language, ending with a JSON block for evaluation
```

- **Graph ownership:** dependencies, owners, escalation, and alerts come only from the
  catalog. The LLM may only extract `INCIDENT`, `SERVICE`, `DATASTORE`, `SYMPTOM`, and
  `MITIGATION` entities with `AFFECTED`, `ROOT_CAUSE_IN`, `SHOWED`, and `RESOLVED_BY`
  relations; anything else is dropped. Root causes are relations, not nodes.
- **Entity alignment:** extracted names are mapped onto catalog names through generated
  and declared aliases (including Chinese ones, such as `支付服务`). Unknown services are
  dropped and reported. Symptoms are mapped to a controlled vocabulary
  (`latency_high`, `error_rate_high`, `consumer_lag`, ...) so that "the same symptom" can
  be matched across incidents. The incident ID comes from the postmortem's front matter.
- **Provenance and deletion:** every extracted edge and entity keeps the `doc_ids` and
  `chunk_ids` that support it. Deleting a postmortem removes only facts no other
  document still supports; catalog facts are never removed by a document deletion.
  Extraction runs before any write, so a failed extraction changes nothing, and a
  failed graph write rolls back the vector insert. Extraction results are cached in
  Redis by chunk text, extraction version, and model.
- **Graph retrieval:** root-cause candidates are ranked by how many symptomatic services
  they lie downstream of, how close they are, and how often they were the root cause of
  past incidents. Up to five graph facts are appended after reranking (reserved slots),
  so the reranker cannot drop them; the passages behind them are fused by RRF.
- **Answers and metrics:** each answer ends with a hidden JSON block
  (`root_cause_services`, `affected_services`, `owner_team`, `uncertain`) for later
  evaluation. The CLI and the web demo report retrieval, first-token, generation, and
  total time, plus LLM calls and estimated tokens.
- **Web demo:** a mode switch (hybrid, graph, hybrid + graph) and a side-by-side
  comparison of hybrid and hybrid + graph. Document evidence and graph reasoning paths
  are shown separately; the JSON block sits in a collapsed "structured result". The
  Documents view adds catalog import and Markdown upload (runbook or postmortem). If
  Neo4j is down or no catalog is imported, questions fall back to hybrid retrieval and
  the page says so; adding and deleting on-call documents need Neo4j.

| Module | Responsibility |
| --- | --- |
| `kb.py` | Add, list, and delete documents and import the catalog, for the CLI and web demo |
| `pdf_ingestion.py` | MinerU parsing, figure descriptions, adding, listing, and deleting PDFs |
| `mineru_content_list.py` | Runs inside MinerU's environment and writes the content list |
| `retrieval.py` | Query rewriting, BM25, rank fusion, and optional metadata scopes |
| `services.py` | Splitting, query engine assembly, and chat memory |
| `base_llm.py` | Qwen, embedding, and reranker factories |
| `storage.py` | MongoDB, Redis, Chroma, and Neo4j connections; `KB_DOMAIN` |
| `graph_schema.py` | Extraction schema and the controlled symptom vocabulary |
| `catalog.py` | Catalog validation, aliases, Neo4j import, and catalog text documents |
| `graph_ingestion.py` | Markdown chunking, extraction, alignment, graph writes, and cleanup |
| `graph_retrieval.py` | Entity linking, Cypher templates, root-cause ranking, the three modes |
| `metrics.py` | Answer timing and token counts |
| `main.py` | CLI: add, delete, list, ask, chat, and import-catalog |
| `app.py` | Streamlit interface and resource caching |
| `ui_text.py` | Chinese and English interface text |
| `prompts.py` | Answer and rewrite prompts, plus reusable template variants |
| `ingestion.py`, `data/` | Sample text policies used only by the offline tests |
| `tests/fixtures/oncall/` | A small catalog, runbook, and postmortems used only by the tests |

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
| `KB_DOMAIN` | `policy` (default) or `oncall`; on-call enables Markdown, the catalog, and Neo4j |
| `NEO4J_PASSWORD` | On-call only: Neo4j password; also used by Compose to create the database |
| `NEO4J_USER`, `NEO4J_URL` | Optional; defaults `neo4j` and `bolt://127.0.0.1:7687` |

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

For the on-call domain, also start Neo4j (with APOC) through the `oncall` profile. Set a
strong `NEO4J_PASSWORD` in `.env` first: Neo4j stores the password when its volume is
created, so changing it later requires changing it inside Neo4j too.

```bash
docker compose --profile oncall up -d --wait
```

Neo4j binds to localhost ports 7474 (browser) and 7687 (Bolt).

**Setting up an on-call knowledge base:** keep the policy settings for the policy
knowledge base, and use a new namespace for on-call documents:

```dotenv
KB_DOMAIN=oncall
KB_NAME=oncall_v1
NEO4J_PASSWORD=choose-a-strong-password
```

A multilingual embedding such as `EMBEDDING_MODEL=BAAI/bge-m3` suits English runbooks
with Chinese questions; change it only together with a new `KB_NAME`. To return to the
policy knowledge base, restore `KB_DOMAIN=policy` and its previous `KB_NAME`.

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

- In [the local demo](http://127.0.0.1:8501), open **Documents**, add a PDF, wait
  for the steps to finish, then switch to **Ask**. Delete asks for confirmation.
  Skipped figures are listed after the upload and counted in the document table.
- `list` shows each document ID, file name, chunk count, upload time, and skipped
  figures. Use the ID with `delete`. Deleted PDFs can be added again.
- Adding the same file twice is rejected before any parsing or model calls.
- Questions are rejected until at least one PDF has been added. Enter `/exit` to leave
  CLI chat; reuse the same session to restore history.
- The web demo rebuilds its search engine after its own uploads and deletions. Restart
  Streamlit after adding or deleting PDFs from the CLI so keyword search sees them.

On-call commands (`KB_DOMAIN=oncall`). Import the catalog first, then runbooks, then
postmortems; postmortems need the catalog for alignment.

```bash
python main.py import-catalog oncall/service_catalog.yaml --alerts oncall/alerts.yaml
python main.py add oncall/runbooks/cart.md
python main.py add oncall/postmortems/INC-001.md --doc-type postmortem
python main.py ask "checkout P99 延迟飙升，同时 payment 错误率上升，根因是什么？"
python main.py ask "If product-catalog is down, which services are affected?" --mode hybrid
python main.py ask "Who owns cart and where do we escalate?" --mode graph
```

- `--mode` is `hybrid`, `graph`, or `hybrid+graph` (default). Without a catalog in
  Neo4j, or when Neo4j is down, questions use `hybrid` and say so.
- Importing the catalog again replaces it: removed dependencies disappear from the graph
  and the catalog's text documents are rebuilt. The catalog row cannot be deleted.
- The Markdown type comes from `--doc-type`, then `doc_type` in the front matter, then a
  `runbooks/` or `postmortems/` folder. Postmortems need `incident_id` in their front
  matter:

```markdown
---
doc_type: postmortem
incident_id: INC-001
title: Payment charge failures
---
# Summary
Checkout error rate rose to 30% because payment rejected charges after a bad config.
# Resolution
We rolled back payment to v2.3.
```

The catalog lists `teams` (with optional `escalates_to`), `services` and `datastores`
(each with `team`, optional `description`, `slo`, `aliases`, and `depends_on` entries of
`target` and `type`: `rpc`, `async`, `cache`, or `db`), and optional `alerts` (`name`,
`service`, `metric`, and `runbook: {file, section}`). Names must be unique across all
of them. See `tests/fixtures/oncall/service_catalog.yaml` for a complete example.

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
Streamlit upload, deletion, streaming, and error behavior. For the on-call domain they
cover catalog validation and aliases, Markdown chunking, extraction alignment and
caching, rollback on failed writes, every Cypher template against a catalog-backed fake,
root-cause ranking, the three modes, and the compare view.

With the three database services running:

```bash
POLICYDESK_INTEGRATION=1 python -m pytest -k database -q
```

This builds and reloads an isolated `policydesk_test_*` namespace from the sample text
policies, verifies metadata filtering through Chroma and BM25, and restores Redis chat
history. It uses test-only models and does not call Qwen or MinerU.

The Neo4j integration test imports the catalog, ingests the fixture postmortems with a
fake extractor, runs every query type, and checks that deletions return the graph to
the catalog alone. It **wipes the graph**, so run it only against a throwaway Neo4j:

```bash
POLICYDESK_INTEGRATION=1 POLICYDESK_NEO4J_WIPE_OK=1 python -m pytest tests/test_graph_database.py -q
```

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

On-call limitations: Neo4j Community has one database, so the graph is not separated by
`KB_NAME`; use one Neo4j per on-call knowledge base. Graph extraction quality depends on
Qwen and has not yet been measured on a real corpus. Token counts use LlamaIndex's
default tokenizer and are estimates; the web demo shares one counter per process.
Answers never execute actions; they only cite runbooks, postmortems, and the catalog.

## Future work

Next on-call milestones are an evaluation script that compares the three modes on the
structured JSON answers, and a realistic runbook and postmortem corpus. Other possible
follow-up improvements include a representative bilingual evaluation set,
Markdown conversion and splitting for large tables, section headings in metadata,
Chinese word segmentation for keyword search, background ingestion jobs, several
companies per deployment with authentication, and explicit version replacement.
