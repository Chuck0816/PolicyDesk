# PolicyDesk

PolicyDesk is a small internal policy assistant with a local web demo and persistent
CLI conversations. It answers questions about a fictional company's eight English
policies and displays the retrieved evidence.

## Implemented architecture

```text
Policy files + validated headers
  -> business metadata -> SentenceSplitter (512 tokens, 50 overlap)
  -> MongoDB nodes + Redis index structure + Chroma embeddings

Question + optional category
  -> one guarded Qwen query rewrite
  -> scoped vector search + scoped BM25
  -> reciprocal rank fusion -> BGE reranking -> compact Qwen answer + excerpts
```

- **Business metadata:** each policy node inherits `policy_title`, `policy_id`,
  `version`, `owner`, `doc_type`, `language`, and `category`, alongside reader
  metadata such as `file_name`. IDs, versions, and owners come from the file header;
  categories come from an explicit filename registry. Missing headers and unregistered
  policy files stop indexing rather than creating guessed labels.
- **Optional scope:** the web selector and CLI `--category` apply
  `category = selected AND doc_type = company_policy`. Chroma filters candidates
  before vector search; BM25 is built from the matching saved nodes. Neither query
  rewriting nor an empty result broadens the selected scope. All-policies search
  also includes a small general HR guidance document.
- **Query preparation:** one Qwen rewrite prepares an English search query while the
  original question remains available for reranking and answer generation. Invalid,
  multiline, oversized, or numerically changed rewrites fall back to the original;
  provider failures also fall back. These guards do not prove semantic equivalence.
- **Hybrid retrieval:** vector and English BM25 searches each return up to eight
  candidates. Reciprocal rank fusion deduplicates evidence and keeps up to ten.
  BM25 is rebuilt in memory from the database's indexed nodes, never from a separate
  local corpus or potentially newer source files.
- **Post-retrieval processing:** local `BAAI/bge-reranker-large` keeps up to four
  excerpts. Displayed scores are reranker relevance scores, not probabilities or
  answer-confidence estimates.
- **Answers and sessions:** Qwen Plus streams English answers using a compact
  context synthesizer and an explicit unknown-answer instruction. CLI chat uses
  question condensation and a Redis-backed 3,000-token memory window. Histories
  are separated by knowledge base, session, and selected category. Empty model
  output raises an actionable error instead of being displayed as success or saved
  as an empty CLI chat turn.
- **Web demo:** independent questions, example prompts, a scope selector, local
  policy previews, streamed answers, and expandable excerpts with policy ID,
  version, category, and relevance score. Models are shared across scope-specific
  engines and load only when the first valid question is submitted.

| Module | Responsibility |
| --- | --- |
| `ingestion.py` | Policy registry, header validation, and document metadata |
| `retrieval.py` | Scope filters, query rewriting, BM25, and rank fusion |
| `services.py` | Splitting, query engine assembly, and chat memory |
| `base_llm.py` | Qwen, embedding, and reranker factories |
| `storage.py` | MongoDB, Redis, and Chroma connections |
| `main.py` | Preview, index build, questions, and persistent CLI chat |
| `app.py` | Streamlit interface and resource caching |
| `prompts.py` | Answer and rewrite prompts, plus reusable template variants |

The vector-only service mode remains available for baseline comparisons and tests.
Both public interfaces use the advanced pipeline.

## Policy library

The files in `data/` contain illustrative rules, amounts, salary bands, and reporting
relationships, not real employee records or actual company secrets.

| File | Category | Coverage |
| --- | --- | --- |
| `leave_policy.txt` | `leave` | Requests, approval thresholds, emergencies, handovers |
| `travel_policy.txt` | `travel` | Approvals, lodging limits, invoices, claim deadlines |
| `overtime_policy.txt` | `overtime` | Working hours, cumulative approvals, compensatory time |
| `salary_policy.txt` | `salary` | Pay dates, L1–L5 bands, reviews, payroll corrections |
| `promotion_policy.txt` | `promotion` | Career levels, evidence, review windows, appeals |
| `employee_benefits_policy.txt` | `benefits` | Learning budgets, wellbeing, equipment claims |
| `information_classification_policy.txt` | `information_security` | Public, Internal, Confidential, Restricted handling |
| `company_hierarchy_policy.txt` | `organization` | Departments, reporting lines, roles, delegation |

New policy files must be registered in `POLICY_CATEGORIES` and start with a title,
`Policy ID:`, `Version:`, and `Policy owner:` (or `Policy owners:`) before the
first blank line. Metadata is descriptive; a version field does not automatically
resolve competing or superseded policies.

## Installation

Requirements: Python 3.11, Docker with Compose, and a DashScope key authorized for
`qwen-plus`. Internet access is needed for dependency installation, Qwen calls,
and initial HuggingFace model downloads. Embedding and reranker weights run locally
and can use substantial memory; initial startup is slower.

Run these commands from the PolicyDesk directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For an existing uv-managed environment without pip, use
`uv pip install --python .venv/bin/python -r requirements.txt` instead.
Development dependencies are in `requirements-dev.txt`.

For a new installation, copy `.env.example` to `.env`. If `.env` already exists,
edit it and preserve existing settings. Obtain a key through the
[official Model Studio API key guide](https://help.aliyun.com/zh/model-studio/get-api-key).

```dotenv
DASHSCOPE_API_KEY=your_dashscope_api_key
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
RERANKER_MODEL=BAAI/bge-reranker-large
KB_NAME=policydesk_metadata_v1
MONGO_DB=policydesk
```

| Variable | Meaning |
| --- | --- |
| `DASHSCOPE_API_KEY` | Required for Qwen model initialization and inference |
| `EMBEDDING_MODEL` | HuggingFace model ID or local directory; default shown above |
| `RERANKER_MODEL` | HuggingFace reranker ID or local directory; default shown above |
| `KB_NAME` | Shared namespace for saved nodes, vectors, index structure, and chat keys |
| `MONGO_DB` | MongoDB database name; default `policydesk` |
| `COMPOSE_PROJECT_NAME` | Optional existing Docker project name; retain it when reusing volumes |

The code loads `.env` beside the Python modules. Shell variables take precedence.
Keep real keys out of source control. The DashScope adapter uses its default
China (Beijing) endpoint, so the key must be valid for that region. The existing
embedding default is retained for index compatibility; switching embeddings requires
a new knowledge base.

Start the services:

```bash
docker compose up -d --wait
docker compose ps
```

MongoDB, Redis, and Chroma bind to localhost ports 27028, 6389, and 8018.
Named volumes persist data. Keep existing Compose project and MongoDB database
settings when reusing an installation.

## Build and run

```bash
python main.py preview
python main.py build
python main.py ask "Who approves four working days of leave?"
python main.py ask "When is monthly salary paid?" --category salary
python main.py chat --session employee-travel --category travel
python -m streamlit run app.py
```

- `preview` shows source text, node counts, and metadata without loading models or
  connecting to databases.
- `build` indexes the entire registered library once. It refuses a populated
  namespace. `--category` applies only to `ask` and `chat`.
- Omit `--category` to search all policies. Enter `/exit` to leave CLI chat.
  Reuse the same session and scope to restore history.
- Open [the local demo](http://127.0.0.1:8501), select a scope, and submit a question.
  Changing scope clears the previous displayed answer without querying automatically.
  Web questions do not share the CLI conversation history.
- The sidebar reads local files, while answers use the saved database snapshot.
  Editing a text file does not update an existing index.

### Existing-index migration

Older indexes do not gain business metadata by changing the loader. To use the new
scopes, choose an unused `KB_NAME` in `.env` (for example,
`policydesk_metadata_v1`), keep the existing `MONGO_DB` and
`COMPOSE_PROJECT_NAME`, then run `python main.py build` and restart Streamlit.

Use another new namespace after changing policy content, metadata, or the embedding
model, or after an interrupted build. Query paths only open existing Chroma
collections; they do not create empty collections. Builds are not transactional
across the three databases, and there is no automatic partial-build repair.
Old namespaces and histories are retained; no migration command deletes them.

## Testing

Install development dependencies, then run the offline suite and code checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Use `uv pip install --python .venv/bin/python -r requirements-dev.txt` if the
environment has no pip. Offline tests use model/storage doubles and need no API
key, model downloads, or running databases. They cover header validation, metadata
inheritance, both retrieval modes' scope isolation, fusion, rewrite fallback,
original-question preservation, conversation memory, collection lifecycle, and
Streamlit streaming/error/scope/cache behavior.

With the three database services running:

```bash
POLICYDESK_INTEGRATION=1 python -m pytest -k database -q
```

This builds and reloads an isolated `policydesk_test_*` namespace, verifies metadata
filtering through Chroma and BM25, and restores Redis chat history. It uses test-only
embedding and language models plus a lightweight postprocessor; it does not call
Qwen or test BGE model quality. Test namespaces are retained, and existing policy
namespaces are not modified.

If Qwen returns no answer, check the key, region, and account quota. In particular,
`AllocationQuota.FreeTierOnly` means that the account has exhausted its free quota
while paid usage is disabled. Account and billing settings must be resolved in Model
Studio; the application does not change them or substitute another provider.

For a real-model smoke test, use `ask` with a built namespace and a valid key, then
inspect the answer and excerpts. For example, ask about payroll within `salary`,
then ask a travel question within that same scope: every returned excerpt must
remain a salary-policy excerpt, and unsupported claims should be reported as
unspecified. This is a manual check, not a measured retrieval-quality benchmark.

## Operations and limitations

```bash
docker compose stop
docker compose start
docker compose logs --tail 30
```

Restart Streamlit after configuration or index changes to refresh cached resources.
Keep Docker volumes to retain data; do not use volume-removal commands for routine
restarts. Stop Streamlit with Ctrl+C.

This remains a simplified local application without user authentication, access
control, incremental indexing, or production monitoring. Category filters are search
controls, not permission checks. Policy excerpts are supporting evidence rather
than verified sentence-level citations. Retrieved context and prompts reduce
unsupported answers but cannot guarantee correctness; unresolved policy questions
should go to the relevant policy owner.

## Future work

Possible follow-up improvements include a representative retrieval/answer evaluation
set, parent–child context expansion when longer documents justify it, and explicit
effective-date/version resolution. Authentication and access control would be needed
before a shared deployment with sensitive records. These are not implemented.
Query rewriting, business metadata filtering, hybrid retrieval, and reranking are
already implemented and are not roadmap items.
