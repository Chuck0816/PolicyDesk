"""Interface text for the web demo in Chinese and English; answers follow the question."""

LANGUAGES = {"zh": "中文", "en": "EN"}
DEFAULT_LANGUAGE = "zh"

TEXT = {
    "zh": {
        "language": "界面语言",
        "nav_ask": "提问",
        "nav_docs": "文档 · {count}",
        "ask_title": "关于公司制度，直接问。",
        "question_label": "你的问题",
        "question_placeholder": "例如：出差住宿标准是多少？ / Who approves four days of leave?",
        "scope": "在 {count} 份文档中搜索 · 用中文或英文提问",
        "ask_button": "提问 →",
        "empty_ask": "还没有文档。先在「文档」页上传公司制度 PDF，然后就可以提问。",
        "go_docs": "去上传 PDF",
        "enter_question": "请先输入问题。",
        "phase_search": "检索制度原文",
        "phase_rank": "重排证据",
        "phase_write": "撰写回答",
        "sources_label": "来源 · {count} 段证据",
        "answer_label": "回答",
        "last_answer_label": "上一个问题的回答",
        "excerpts": "查看原文摘录 · 相关度为重排模型的相对分数，不代表答案的可信度",
        "no_sources": "没有找到相关的原文。",
        "page": "第 {page} 页",
        "relevance": "相关度 {score}",
        "kind_text": "正文",
        "kind_table": "表格",
        "kind_figure": "图片",
        "examples_label": "试试这些问题",
        "examples": [
            "出差住宿标准是多少？",
            "报销申请需要在多少天内提交？",
            "Who approves business class flights?",
        ],
        "disclaimer": "回答仅依据检索到的制度原文。未解决的问题请联系 HR 或财务。",
        "question_failed": "无法完成提问。{error}",
        "query_failed": (
            "无法完成提问。请检查数据库服务、API key 和索引，详情见终端输出，然后重试。"
        ),
        "docs_title": "文档库",
        "docs_intro": "上传公司制度 PDF。表格和图片会被解析，员工即可用中文或英文提问。",
        "upload_label": "选择 PDF",
        "upload_hint": "同一份文件不会重复添加 · 大文件解析可能需要几分钟",
        "add_button": "加入知识库",
        "adding": "正在添加 {name}",
        "adding_hint": "大文件可能需要几分钟",
        "step_upload": "上传",
        "step_parse": "解析版面与表格",
        "step_describe": "描述图片",
        "step_index": "写入知识库",
        "added": "已添加 {name}（{count} 段）。",
        "skipped": "有 {count} 张图片无法生成文字描述，已跳过，其内容暂不可检索；其余内容已入库。",
        "skipped_item": "第 {page} 页{caption} — {reason}",
        "add_failed": "无法添加 {name}。{error}",
        "filter_all": "全部 {count}",
        "filter_skipped": "有跳过图片 {count}",
        "col_file": "文件",
        "col_status": "状态",
        "col_chunks": "段数",
        "col_added": "添加时间",
        "status_ok": "可检索",
        "status_skipped": "{count} 张图片已跳过",
        "delete": "删除",
        "cancel": "取消",
        "confirm_delete": "确定从知识库删除 {name}？删除后员工将无法再搜到它，可以重新上传。",
        "deleted": "已删除 {name}。",
        "delete_failed": "无法删除 {name}。{error}",
        "no_docs": "还没有文档。上传第一份 PDF 开始使用。",
        "no_skipped": "没有跳过图片的文档。",
        "docs_foot": (
            "“段数”是文档切分后用于检索的片段数量。跳过的图片可以在修复问题后删除文档并重新上传。"
        ),
        "db_error": "无法连接数据库服务。请用 Docker 启动服务后刷新页面。",
        "list_unavailable": "文档列表暂不可用。",
    },
    "en": {
        "language": "Language",
        "nav_ask": "Ask",
        "nav_docs": "Documents · {count}",
        "ask_title": "Ask about company policy.",
        "question_label": "Your question",
        "question_placeholder": (
            "For example: Who approves four days of leave? / 出差住宿标准是多少？"
        ),
        "scope": "Searching {count} documents · Ask in Chinese or English",
        "ask_button": "Ask →",
        "empty_ask": "No documents yet. Add policy PDFs on the Documents page, then ask.",
        "go_docs": "Add PDFs",
        "enter_question": "Enter a question first.",
        "phase_search": "Search policies",
        "phase_rank": "Rank evidence",
        "phase_write": "Write answer",
        "sources_label": "Sources · {count} excerpts",
        "answer_label": "Answer",
        "last_answer_label": "Answer to your last question",
        "excerpts": (
            "Original excerpts · scores are relative reranker scores, not answer confidence"
        ),
        "no_sources": "No matching excerpts were found.",
        "page": "p. {page}",
        "relevance": "Relevance {score}",
        "kind_text": "Text",
        "kind_table": "Table",
        "kind_figure": "Figure",
        "examples_label": "Try asking",
        "examples": [
            "What is the hotel limit for business travel?",
            "How many days do I have to submit an expense claim?",
            "请假 4 个工作日需要谁审批？",
        ],
        "disclaimer": (
            "Answers use only the retrieved policy text. "
            "Contact HR or Finance for anything unresolved."
        ),
        "question_failed": "Could not complete the question. {error}",
        "query_failed": (
            "Could not complete the question. Check the database services, API key, "
            "and saved index. See the terminal for details, then retry."
        ),
        "docs_title": "Documents",
        "docs_intro": (
            "Add your company's policy PDFs. Tables and figures are parsed so employees "
            "can ask in Chinese or English."
        ),
        "upload_label": "Choose a PDF",
        "upload_hint": "The same file is never added twice · large files can take minutes",
        "add_button": "Add to knowledge base",
        "adding": "Adding {name}",
        "adding_hint": "Large files can take a few minutes",
        "step_upload": "Upload",
        "step_parse": "Parse layout and tables",
        "step_describe": "Describe figures",
        "step_index": "Save to knowledge base",
        "added": "Added {name} ({count} chunks).",
        "skipped": (
            "{count} figure(s) could not be described and were skipped; their content is "
            "not searchable. The rest of the PDF was added."
        ),
        "skipped_item": "Page {page}{caption} — {reason}",
        "add_failed": "Could not add {name}. {error}",
        "filter_all": "All {count}",
        "filter_skipped": "With skipped figures {count}",
        "col_file": "File",
        "col_status": "Status",
        "col_chunks": "Chunks",
        "col_added": "Added",
        "status_ok": "Searchable",
        "status_skipped": "{count} figure(s) skipped",
        "delete": "Delete",
        "cancel": "Cancel",
        "confirm_delete": (
            "Delete {name} from the knowledge base? Employees will no longer find it; "
            "you can add it again later."
        ),
        "deleted": "Deleted {name}.",
        "delete_failed": "Could not delete {name}. {error}",
        "no_docs": "No documents yet. Add your first PDF to get started.",
        "no_skipped": "No documents have skipped figures.",
        "docs_foot": (
            "Chunks are the passages a document is split into for search. To recover a "
            "skipped figure, fix the cause, delete the document, and add it again."
        ),
        "db_error": "Could not reach the database services. Start them with Docker, then reload.",
        "list_unavailable": "The document list is unavailable.",
    },
}


# Keys shared by both domains that only the on-call screens use.
SHARED = {
    "zh": {
        "mode_label": "检索模式",
        "mode_hybrid": "混合检索",
        "mode_graph": "图检索",
        "mode_hybrid+graph": "混合 + 图",
        "compare": "并排对比（混合 vs 混合 + 图）",
        "graph_unavailable": "图检索暂不可用（{reason}），将使用混合检索。",
        "graph_paths_label": "图推理路径 · {count} 条",
        "structured_label": "结构化结果（用于评测）",
        "timing": "检索 {retrieval} 秒 · 首字 {first} 秒 · 生成 {generation} 秒 · 共 {total} 秒 · "
        "LLM 调用 {calls} 次 · 约 {tokens} tokens（估算）",
        "step_extract": "抽取图谱",
        "md_type": "Markdown 类型",
        "md_auto": "自动识别",
        "md_runbook": "Runbook",
        "md_postmortem": "故障复盘",
        "catalog_title": "服务目录",
        "catalog_status": "已导入 {services} 个服务，{alerts} 条告警。",
        "catalog_missing": "还没有导入服务目录。先导入它，再添加故障复盘。",
        "catalog_alerts": "告警定义（可选）",
        "catalog_button": "导入服务目录",
        "catalog_imported": "已导入服务目录：{services} 个服务，{teams} 个团队，{alerts} 条告警。",
        "catalog_failed": "无法导入服务目录。{error}",
        "status_relations": "图谱 {count} 条关系",
        "status_services": "{count} 个服务",
        "type_pdf": "PDF",
        "type_runbook": "Runbook",
        "type_postmortem": "故障复盘",
        "type_service_catalog": "服务目录",
        "graph_report": "图谱抽取：{relations} 条关系；{empty} 个片段没有抽到事实；"
        "未对齐而丢弃：{dropped}",
        "needs_graph": "On-call 知识库的增删需要连接 Neo4j：{reason}",
    },
    "en": {
        "mode_label": "Retrieval mode",
        "mode_hybrid": "Hybrid",
        "mode_graph": "Graph",
        "mode_hybrid+graph": "Hybrid + graph",
        "compare": "Compare side by side (hybrid vs hybrid + graph)",
        "graph_unavailable": "Graph retrieval is unavailable ({reason}); using hybrid retrieval.",
        "graph_paths_label": "Graph reasoning · {count}",
        "structured_label": "Structured result (for evaluation)",
        "timing": "Retrieval {retrieval}s · first token {first}s · generation {generation}s · "
        "total {total}s · {calls} LLM calls · ~{tokens} tokens (estimated)",
        "step_extract": "Extract graph",
        "md_type": "Markdown type",
        "md_auto": "Detect",
        "md_runbook": "Runbook",
        "md_postmortem": "Postmortem",
        "catalog_title": "Service catalog",
        "catalog_status": "{services} services and {alerts} alerts imported.",
        "catalog_missing": "No service catalog yet. Import it before adding postmortems.",
        "catalog_alerts": "Alert definitions (optional)",
        "catalog_button": "Import catalog",
        "catalog_imported": "Imported the catalog: {services} services, {teams} teams, "
        "{alerts} alerts.",
        "catalog_failed": "Could not import the catalog. {error}",
        "status_relations": "{count} graph relations",
        "status_services": "{count} services",
        "type_pdf": "PDF",
        "type_runbook": "Runbook",
        "type_postmortem": "Postmortem",
        "type_service_catalog": "Service catalog",
        "graph_report": "Graph extraction: {relations} relations; {empty} chunk(s) without "
        "facts; dropped as not in the catalog: {dropped}",
        "needs_graph": "Changing an on-call knowledge base needs Neo4j: {reason}",
    },
}

# On-call wording that replaces the policy wording for the same keys.
ONCALL = {
    "zh": {
        "ask_title": "值班排障，直接问。",
        "phase_search": "检索文档与图谱",
        "go_docs": "去导入服务目录",
        "upload_label": "选择 Markdown 或 PDF",
        "no_docs": "还没有文档。先导入服务目录，再添加 runbook 和故障复盘。",
        "question_placeholder": (
            "例如：checkout P99 延迟飙升，同时 payment 错误率上升，根因是什么？"
        ),
        "scope": "在 {count} 份文档和服务依赖图中搜索 · 用中文或英文提问",
        "empty_ask": "还没有文档。先在「文档」页导入服务目录，再添加 runbook 和故障复盘。",
        "examples": [
            "如果 product-catalog 不可用，会影响哪些服务？",
            "checkout 延迟升高，同时 payment 错误率上升，最可能的根因是什么？",
            "Who owns the currency service and what is the escalation path?",
        ],
        "disclaimer": (
            "回答只依据 runbook、故障复盘和服务目录；不会执行任何操作。不确定时请升级给负责团队。"
        ),
        "docs_intro": "导入服务目录，并上传 runbook 和故障复盘（Markdown），也可以上传 PDF。",
        "upload_hint": "PDF 或 Markdown · 同一份文件不会重复添加 · 故障复盘会抽取图谱",
    },
    "en": {
        "ask_title": "On-call troubleshooting, just ask.",
        "phase_search": "Search documents and graph",
        "go_docs": "Import the catalog",
        "upload_label": "Choose a Markdown file or PDF",
        "no_docs": "No documents yet. Import the service catalog, then add runbooks and "
        "postmortems.",
        "question_placeholder": "For example: checkout p99 latency is up and payment errors "
        "are rising. What is the likely root cause?",
        "scope": "Searching {count} documents and the service graph · Chinese or English",
        "empty_ask": "No documents yet. Import the service catalog on the Documents page, "
        "then add runbooks and postmortems.",
        "examples": [
            "If product-catalog is unavailable, which services are affected?",
            "checkout 延迟升高，同时 payment 错误率上升，最可能的根因是什么？",
            "Who owns the currency service and what is the escalation path?",
        ],
        "disclaimer": "Answers use only runbooks, postmortems and the service catalog; nothing "
        "is executed. When unsure, escalate to the owning team.",
        "docs_intro": "Import the service catalog and add runbooks and postmortems "
        "(Markdown); PDFs work too.",
        "upload_hint": "PDF or Markdown · the same file is never added twice · "
        "postmortems are added to the graph",
    },
}


def translate(language, key, domain="policy", **values):
    """Return interface text in the chosen language and domain, filled with the values."""
    language = language if language in TEXT else DEFAULT_LANGUAGE
    if domain == "oncall" and key in ONCALL[language]:
        text = ONCALL[language][key]
    elif key in SHARED[language]:
        text = SHARED[language][key]
    else:
        text = TEXT[language][key]
    return text.format(**values) if values else text
