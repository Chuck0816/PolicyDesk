"""PDF ingestion: MinerU blocks + Qwen-VL figure text, appended to the saved index."""

import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import NodeRelationship, ObjectType, RelatedNodeInfo

import services
import storage

LOGGER = logging.getLogger(__name__)
MINERU_COMMAND = os.getenv("MINERU_COMMAND", "mineru")
# Optional: the Python of MinerU's environment; by default the one beside MINERU_COMMAND.
MINERU_PYTHON = os.getenv("MINERU_PYTHON", "")
# MinerU 4.x has no content-list command, so this script writes one in MinerU's environment.
MINERU_ADAPTER = Path(__file__).resolve().with_name("mineru_content_list.py")
MISSING_MINERU = (
    "MinerU was not found ({command!r}). Install MinerU in its own environment, "
    "set MINERU_COMMAND in .env to the full path of its `mineru` executable, "
    "then restart PolicyDesk."
)
# 给 PDF 里的图片生成文字描述时用哪个视觉语言模型（VLM）。默认是 DashScope 的 qwen3-vl-plus
VL_MODEL = os.getenv("VL_MODEL", "qwen3-vl-plus")
UPLOAD_DOC_TYPE = "uploaded_document"
# Redis keeps the whole index structure as one value, so writes must not overlap.
WRITE_LOCK = threading.Lock()

SKIPPED = {"header", "footer", "page_number"}
FIGURE_PROMPT = (
    "Describe this image from a company policy PDF so employees can search it. "
    "For flowcharts, list every step, role, and condition in order. "
    "Copy every number exactly and do not add facts. Caption: {caption}"
)

"""run_mineru 会拼出一条 MinerU 命令，用子进程运行它来解析 PDF, 然后在输出目录里找到唯一的 content_list.json, 返回它的路径"""


def run_mineru(pdf, output_dir):
    """
    subprocess.run(列表, check=True) -> run a subprocess using the command
    *shlex.split(MINERU_COMMAND), "-p", str(pdf), "-o", str(output_dir)
    shlex.split(str) -> 按照空格拆分str
    pdf -> Path对象
    输出["mineru", "-p", "/docs/差旅政策.pdf", "-o", "/tmp/abc"]命令行参数
    """
    # MinerU 4.x rejects `mineru -p ... -o ...`; the same arguments now go to MINERU_ADAPTER,
    # run by MinerU's Python: [python, mineru_content_list.py, "-p", pdf, "-o", output_dir].
    command = [mineru_python(), str(MINERU_ADAPTER), "-p", str(pdf), "-o", str(output_dir)]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise ValueError(MISSING_MINERU.format(command=command[0])) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"MinerU stopped with exit code {error.returncode} while parsing {pdf.name}. "
            "See the terminal output for its error."
        ) from error
    # 查找json文件 .rglob(模式)->递归模式
    # pdf.stem
    # 文件名去掉扩展名的部分：
    matches = list(Path(output_dir).rglob(f"{pdf.stem}_content_list.json"))

    if len(matches) != 1:
        raise ValueError(f"Expected one MinerU content list, found {len(matches)}.")
    return matches[0]


def mineru_python():
    """Return MINERU_PYTHON, or the `python` in the same folder as the MINERU_COMMAND executable."""
    if MINERU_PYTHON:
        return MINERU_PYTHON
    name = shlex.split(MINERU_COMMAND)[0]
    executable = shutil.which(name)
    if executable is None:
        raise ValueError(MISSING_MINERU.format(command=name))
    return str(Path(executable).parent / "python")


"""这个函数把一张图片和它的图注发给 Qwen-VL,
让模型把图片内容写成一段可以被检索的文字，然后返回这段文字"""


def describe_image(image_path, caption):
    from dashscope import MultiModalConversation

    response = MultiModalConversation.call(
        model=VL_MODEL,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": Path(image_path).resolve().as_uri()},  # 转化成绝对路径后写成url格式
                    # caption是图注，比如 图1 差旅报销审批流程 <-（图注）
                    {"text": FIGURE_PROMPT.format(caption=caption or "-")},
                ],
            }
        ],
    )
    if response.status_code != 200:
        raise RuntimeError(f"{VL_MODEL} failed: {response.code} {response.message}")
    parts = response.output.choices[0].message.content
    return "".join(part.get("text", "") for part in parts).strip()


"""
content_to_blocks: 把 MinerU 的 content_list.json 转成有序的 (类型, 文本, 页码) 块列表.

核心机制: buffer (正文缓冲区)
- buffer 暂存"当前页上连续的正文", buffer_page 记录这些正文属于哪一页.
- flush(): 如果 buffer 不为空, 就把里面的内容用 "\n\n" 拼成一个 text 块输出, 然后清空 buffer;
  如果 buffer 为空, 什么都不做.
- text 块只能通过 flush 生成; table / figure 块不经过 buffer, 直接加入结果.

逐块处理规则:
1. header / footer / page_number → 直接丢弃
2. 换页 (page != buffer_page) → flush, 再更新 buffer_page   ⇒ 正文不会跨页
3. 有 table_body 的表格 → flush, 然后单独生成 table 块 (使用 HTML, 不调用 VLM)
4. 图片, 或只有截图的表格 → flush, 然后调用 VLM 描述, 单独生成 figure 块
5. 其他非空文本 (正文, 标题, 列表, 公式…) → 放入 buffer, 等待合并
6. 遍历结束 → 最后再 flush 一次, 避免最后一页的正文丢失

切分结果:
- 触发 flush 的三个时机: 换页, 遇到表格, 遇到图片 (再加上遍历结束)
- text 块 = 同一页上, 没有被表格或图片打断的连续正文
- 每个表格, 每张图片各自单独成为一个块, 相邻的也不会合并; 块的顺序和原文一致
- 块总数 = 非空 flush 的次数 + 表格数 + 图片数
- 注意: 这里只是按"页面和类型"划分边界; 按长度切分 (SentenceSplitter 512/50) 是在下一步做的
"""


def content_to_blocks(content_list_path, describe=describe_image, skipped=None):
    path = Path(content_list_path)
    blocks, buffer, buffer_page = [], [], None

    def flush():
        if buffer:
            blocks.append(("text", "\n\n".join(buffer), buffer_page))
            buffer.clear()

    for item in json.loads(path.read_text(encoding="utf-8")):
        kind, page = item.get("type"), item.get("page_idx", 0) + 1
        if kind in SKIPPED:
            continue
        if page != buffer_page:  # 换页输出正文
            flush()
            buffer_page = page
        if kind == "table" and item.get("table_body"):  # HTML表格 ->直接使用
            flush()
            caption = " ".join(item.get("table_caption") or [])
            blocks.append(("table", f"Table: {caption}\n{item['table_body']}", page))
        elif kind in ("image", "table", "chart") and item.get("img_path"):  # 调用VLM
            flush()
            caption = " ".join(item.get(f"{kind}_caption") or item.get("img_caption") or [])
            # A failed figure is skipped and reported; the rest of the PDF is still indexed.
            try:
                text = describe(path.parent / item["img_path"], caption)  # vlm
                reason = "" if text else "the model returned no description"
            except Exception as error:
                text, reason = "", f"{type(error).__name__}: {error}"
            if reason:
                LOGGER.warning("Skipped figure on page %s: %s", page, reason)
                if skipped is not None:
                    skipped.append({"page": page, "caption": caption, "reason": reason})
                continue
            blocks.append(("figure", f"Figure: {caption}\n{text}", page))
        elif text := item_text(item):
            buffer.append(text)
    flush()
    return blocks


def item_text(item):
    """Plain text of a non-visual block: paragraphs, titles, lists, code, and equations."""
    parts = [item.get("text"), "\n".join(item.get("list_items") or []), item.get("code_body")]
    return "\n".join(part.strip() for part in parts if part and part.strip())


def ingest_pdf(
    pdf_path, context, collection, *, work_dir, parse=run_mineru, describe=describe_image
):
    """Step 4: dedupe, build documents, reuse the existing splitter, append to the index.

    Returns (doc_id, node count, skipped figures). Each skipped figure is a dict with
    page, caption, and reason. Nothing is written when an exception is raised before
    the index update.
    """
    pdf = Path(pdf_path).resolve()
    if pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files are supported: {pdf.name}")
    # SHA-256是哈希算法, 读出 PDF 的全部字节，算出 SHA-256，取前 16 个字符，拼成 pdf-3f9a1c... 这样的 doc_id。
    doc_id = "pdf-" + hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]
    if context.docstore.ref_doc_exists(doc_id):
        raise ValueError(f"{pdf.name} is already in the knowledge base ({doc_id}).")

    skipped = []
    blocks = content_to_blocks(parse(pdf, Path(work_dir)), describe, skipped)
    if not blocks:
        raise ValueError(
            f"No text, tables, or figures could be extracted from {pdf.name} "
            f"({len(skipped)} figure(s) failed). Nothing was added."
        )
    uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    documents = [
        Document(
            text=text,
            metadata={
                "file_name": pdf.name,
                "policy_id": doc_id,
                "version": "1.0",
                "category": "uploaded",
                "doc_type": UPLOAD_DOC_TYPE,
                "content_type": kind,
                "page": page,
                "uploaded_at": uploaded_at,
                "skipped_figures": len(skipped),
            },
            excluded_embed_metadata_keys=["page", "uploaded_at", "skipped_figures"],
            excluded_llm_metadata_keys=["uploaded_at", "skipped_figures"],
        )
        for kind, text, page in blocks
    ]
    nodes = services.split_documents(documents)
    # 把所有节点归到同一个 doc_id 下
    source = RelatedNodeInfo(node_id=doc_id, node_type=ObjectType.DOCUMENT)
    for node in nodes:
        node.relationships[NodeRelationship.SOURCE] = source  # one doc_id per PDF

    """
    写入检索
    Case A: 已有索引结构（正常情况）
    """
    with WRITE_LOCK:
        # Another upload of the same file may have finished while this one was parsing.
        if context.docstore.ref_doc_exists(doc_id):
            raise ValueError(f"{pdf.name} is already in the knowledge base ({doc_id}).")
        if context.index_store.get_index_struct(storage.INDEX_ID):
            # Required: otherwise nodes only reach Chroma and BM25 never sees them.
            index = load_index_from_storage(
                context, index_id=storage.INDEX_ID, store_nodes_override=True
            )
            # 对于新的节点，向量写入chroma, 元数据写入mango，index写入redis
            index.insert_nodes(nodes)
        else:
            if collection.count():
                raise ValueError("Chroma has data but no index structure; use a new KB_NAME.")
            index = VectorStoreIndex(nodes, storage_context=context, store_nodes_override=True)
            index.set_index_id(storage.INDEX_ID)
    return doc_id, len(nodes), skipped


def list_pdfs(docstore):
    """Return one row per uploaded PDF, newest first, from the docstore's ref-doc records."""
    rows = []
    for doc_id, info in (docstore.get_all_ref_doc_info() or {}).items():
        metadata = info.metadata or {}
        if metadata.get("doc_type") != UPLOAD_DOC_TYPE:
            continue
        rows.append(
            {
                "doc_id": doc_id,
                "file_name": metadata.get("file_name", doc_id),
                "uploaded_at": metadata.get("uploaded_at", ""),
                "skipped_figures": metadata.get("skipped_figures", 0),
                "chunks": len(info.node_ids),
            }
        )
    return sorted(rows, key=lambda row: row["uploaded_at"], reverse=True)


def delete_pdf(context, doc_id):
    """Remove one uploaded PDF from Chroma, Redis, and MongoDB together.

    Settings.embed_model must be configured, because loading the index resolves it.
    """
    with WRITE_LOCK:
        if not context.docstore.ref_doc_exists(doc_id):
            raise ValueError(f"{doc_id} is not in the knowledge base.")
        # Same reason as ingestion: without the override, Redis and MongoDB keep the nodes.
        index = load_index_from_storage(
            context, index_id=storage.INDEX_ID, store_nodes_override=True
        )
        index.delete_ref_doc(doc_id, delete_from_docstore=True)
