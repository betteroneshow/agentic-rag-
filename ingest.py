# -*- coding: utf-8 -*-
"""
@desc: 数据注入脚本（已升级为并行与批处理模式）

本脚本使用多进程并行处理文档，并通过批处理方式存入数据库，以提升注入效率。
"""

import os
import shutil
import multiprocessing
import hashlib
import argparse
import json
from collections import Counter
from tqdm import tqdm
import pandas as pd
import chromadb
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 在加载其他模块前，先加载配置，确保环境变量等设置生效
import config
from agentic_rag.chains import get_embedding_function, get_summarizer_chain
from agentic_rag.ocr import SUPPORTED_IMAGE_SUFFIXES, ocr_image_file, ocr_pdf_file
from config import (
    DOCUMENT_CHUNK_OVERLAP,
    DOCUMENT_CHUNK_SIZE,
    RETRIEVAL_CHUNK_OVERLAP,
    RETRIEVAL_CHUNK_SIZE,
    EXCEL_METADATA_COLUMNS,
)

# --- 配置 ---
DATA_PATH = "data"
PERSIST_PATH = "chroma_db"
SUMMARY_COLLECTION_NAME = "doc_summaries"
CHUNK_COLLECTION_NAME = "doc_chunks"
RETRIEVAL_COLLECTION_NAME = "retrieval_chunks"
MANIFEST_PATH = os.path.join(PERSIST_PATH, "ingestion_manifest.json")
PIPELINE_SIGNATURE = "ocr-parent-child-recursive-v2"


def _manifest_signature() -> str:
    """标识会影响向量兼容性的入库配置。"""
    raw = "|".join([
        PIPELINE_SIGNATURE,
        config.EMBEDDING_PROVIDER,
        config.EMBEDDING_MODEL_NAME,
        str(DOCUMENT_CHUNK_SIZE),
        str(DOCUMENT_CHUNK_OVERLAP),
        str(RETRIEVAL_CHUNK_SIZE),
        str(RETRIEVAL_CHUNK_OVERLAP),
        config.OCR_MODEL_NAME,
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_fingerprint(filepath: str) -> str:
    """用大小和纳秒级修改时间快速判断文件是否发生变化。"""
    stat = os.stat(filepath)
    raw = f"{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        return {"pipeline_signature": _manifest_signature(), "files": {}}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("pipeline_signature") != _manifest_signature():
        raise RuntimeError(
            "入库模型或分块配置已变化，不能与旧索引安全混用。"
            "请执行 python ingest.py --rebuild。"
        )
    manifest.setdefault("files", {})
    return manifest


def _save_manifest(manifest: dict) -> None:
    """原子写入清单，避免程序中断留下半个 JSON 文件。"""
    os.makedirs(PERSIST_PATH, exist_ok=True)
    temp_path = MANIFEST_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, MANIFEST_PATH)


def _delete_stale_records(
    collection,
    current_ids: list[str],
    current_metadatas: list[dict],
    completed_files: set[str],
) -> None:
    """删除已修改文件在本次入库后不再存在的旧记录。"""
    ids_by_file: dict[str, set[str]] = {}
    for item_id, metadata in zip(current_ids, current_metadatas):
        source_file = metadata.get("source_file")
        if source_file in completed_files:
            ids_by_file.setdefault(source_file, set()).add(item_id)

    for source_file, new_ids in ids_by_file.items():
        existing = collection.get(
            where={"source_file": {"$eq": source_file}},
            include=[],
        )
        stale_ids = set(existing.get("ids", [])) - new_ids
        if stale_ids:
            collection.delete(ids=sorted(stale_ids))

TABULAR_FIELD_LABELS = {
    "doc_id": "资料编号",
    "doc_type": "资料类型",
    "region": "地区",
    "destination": "目的地",
    "full_destination": "完整目的地",
    "attraction": "景点",
    "transportation": "交通",
    "food": "美食",
    "accommodation": "住宿",
    "tips": "注意事项",
    "source_row": "来源行",
}
TABULAR_AGGREGATE_MAX_CHARS = 6000

# RecursiveCharacterTextSplitter 会依次尝试这些分隔符：先尽量保持段落，
# 段落过大时再退到句子、分句、词，最后才按字符强制切分。
DOCUMENT_SEPARATORS = [
    "\n\n",  # 段落
    "\n",    # 单行
    "。",     # 中文句号
    "！",     # 中文感叹句
    "？",     # 中文疑问句
    ". ",    # 英文句子
    "! ",
    "? ",
    "；",     # 中文分句
    "; ",
    "，",     # 中文短分句
    ", ",
    " ",      # 英文单词边界
    "",       # 最终按字符切分
]


def create_document_splitter() -> RecursiveCharacterTextSplitter:
    """创建面向中英文叙事文档的层级递归字符切分器。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=DOCUMENT_CHUNK_SIZE,
        chunk_overlap=DOCUMENT_CHUNK_OVERLAP,
        separators=DOCUMENT_SEPARATORS,
        keep_separator=True,
        length_function=len,
        is_separator_regex=False,
    )

def create_retrieval_splitter() -> RecursiveCharacterTextSplitter:
    """创建用于高精度召回的小块分割器。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=RETRIEVAL_CHUNK_SIZE,
        chunk_overlap=RETRIEVAL_CHUNK_OVERLAP,
        separators=DOCUMENT_SEPARATORS,
        keep_separator=True,
        length_function=len,
        is_separator_regex=False,
    )


def _clean_cell(value) -> str:
    """清理表格单元格，过滤空值和常见的无意义占位符。"""
    if pd.isna(value):
        return ""
    value_str = str(value).strip()
    if value_str.lower() in {"", "nan", "none", "null", "nat"}:
        return ""
    return value_str


def _build_tabular_text(row: pd.Series) -> str:
    """将一行表格转换为适合中文语义检索的结构化文本。"""
    lines = ["文档类型：旅游知识表格记录"]
    for column, value in row.items():
        value_str = _clean_cell(value)
        if not value_str:
            continue
        label = TABULAR_FIELD_LABELS.get(str(column), str(column))
        lines.append(f"{label}：{value_str}")
    return "\n".join(lines)


def _tabular_metadata(file_path: str, row: pd.Series, row_index: int) -> dict:
    """构建可供 Chroma 过滤的行级元数据。"""
    metadata = {
        "source": file_path,
        "row_index": int(row_index),
        "record_id": f"row_{row_index}",
        "record_level": "row",
        "data_type": "tabular",
    }
    for column in EXCEL_METADATA_COLUMNS:
        if column not in row.index:
            continue
        value_str = _clean_cell(row[column])
        if value_str:
            metadata[column] = value_str
    return metadata


def _aggregate_record_id(*parts: str) -> str:
    raw = "|".join(str(part) for part in parts)
    return "aggregate_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _build_tabular_aggregates(df: pd.DataFrame, file_path: str) -> list[Document]:
    """按目的地和资料类型生成聚合文档，支持综合行程类查询。"""
    destination_column = next(
        (column for column in ("destination", "full_destination", "region") if column in df.columns),
        None,
    )
    if not destination_column:
        return []

    grouped_rows: dict[tuple[str, str], list[str]] = {}
    for _, row in df.iterrows():
        destination = _clean_cell(row.get(destination_column))
        if not destination:
            continue
        doc_type = _clean_cell(row.get("doc_type")) or "综合"
        grouped_rows.setdefault((destination, doc_type), []).append(_build_tabular_text(row))

    documents: list[Document] = []
    for (destination, doc_type), row_texts in grouped_rows.items():
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_chars = 0
        for row_text in row_texts:
            if current_batch and current_chars + len(row_text) > TABULAR_AGGREGATE_MAX_CHARS:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(row_text)
            current_chars += len(row_text)
        if current_batch:
            batches.append(current_batch)

        for batch_index, batch in enumerate(batches):
            content = (
                f"目的地：{destination}\n"
                f"资料主题：{doc_type}\n"
                "以下是该目的地的相关旅游资料：\n\n"
                + "\n\n".join(batch)
            )
            metadata = {
                "source": file_path,
                "data_type": "tabular",
                "record_level": "aggregate",
                "record_id": _aggregate_record_id(destination, doc_type, str(batch_index)),
                "destination": destination,
                "doc_type": doc_type,
            }
            documents.append(Document(page_content=content, metadata=metadata))
    return documents

# --- 工作函数：用于并行处理 ---
def process_document_worker(doc):
    """
    对单个文档进行摘要生成和文本切分的工作函数。
    注意：为了避免多进程中的序列化问题，此函数内部会自行初始化所需的链和分割器。
    """
    summarizer_chain = get_summarizer_chain()
    text_splitter = create_document_splitter()
    retrieval_splitter = create_retrieval_splitter()
    
    doc_content = doc.page_content
    doc_source = doc.metadata.get('source', 'unknown_source')
    if doc.metadata.get("record_id"):
        doc_source = f"{doc_source}_{doc.metadata['record_id']}"
    elif 'row_index' in doc.metadata:
        doc_source = f"{doc_source}_row_{doc.metadata['row_index']}"
    elif 'page' in doc.metadata:
        doc_source = f"{doc_source}_page_{doc.metadata['page']}"

    try:
        # 1. 根据数据类型智能生成摘要
        doc_type = doc.metadata.get('data_type', 'narrative') # 默认为叙事型
        summary = ""
        if doc_type == 'narrative':
            # 对叙事型文档，调用LLM生成摘要
            summary = summarizer_chain.invoke({"document_content": doc_content}).content
        elif doc_type == 'tabular':
            # 对表格型数据，直接使用原文作为摘要，免除LLM调用
            summary = doc_content
        
        summary_metadata = dict(doc.metadata)
        summary_metadata["source"] = doc_source

        # 表格行和聚合记录必须保持结构完整；叙事文档继续递归字符切分。
        if doc_type == 'tabular':
            splits = [doc]
        else:
            splits = text_splitter.split_documents([doc])
        for split in splits:
            split.metadata["source"] = doc_source
        chunk_ids = [f"{doc_source}_chunk_{i}" for i in range(len(splits))]
        chunk_docs = [split.page_content for split in splits]
        chunk_metadatas = []
        retrieval_ids, retrieval_docs, retrieval_metadatas = [], [], []
        for parent_index, (parent_id, split) in enumerate(zip(chunk_ids, splits)):
            parent_metadata = dict(split.metadata)
            parent_metadata.update({"parent_id": parent_id, "chunk_level": "parent"})
            chunk_metadatas.append(parent_metadata)
            children = [split] if doc_type == "tabular" else retrieval_splitter.split_documents([split])
            for child_index, child in enumerate(children):
                child_id = f"{parent_id}_child_{child_index}"
                child_metadata = dict(child.metadata)
                child_metadata.update({
                    "parent_id": parent_id, "parent_index": parent_index,
                    "chunk_level": "child", "source": doc_source,
                })
                retrieval_ids.append(child_id)
                retrieval_docs.append(child.page_content)
                retrieval_metadatas.append(child_metadata)

        return (doc_source, summary, summary_metadata, chunk_ids, chunk_docs, chunk_metadatas,
                retrieval_ids, retrieval_docs, retrieval_metadatas)
    except Exception as e:
        print(f"处理文档 {doc_source} 时出错: {e}")
        return None

# --- 主逻辑 ---
def main(rebuild: bool = False):
    """
    主函数：执行并行化和批处理的数据注入流程。
    """
    print("---" + " 开始并行化数据注入流程" + " ---")

    # 在删除旧索引、执行 OCR 和摘要生成之前验证 Embedding 权限。
    # 这样模型未开通或端点配置错误时可以快速失败，避免浪费时间和费用。
    print("--- 验证嵌入模型连接与权限 ---")
    try:
        embedding_function = get_embedding_function()
        test_vector = embedding_function.embed_query("旅游知识库连接测试")
        if not test_vector:
            raise RuntimeError("嵌入模型返回了空向量")
        print(f"--- 嵌入模型验证成功，向量维度: {len(test_vector)} ---")
    except Exception as exc:
        raise RuntimeError(
            "嵌入模型预检失败。请确认 EMBEDDING_MODEL_NAME 已开通，并检查 "
            "EMBEDDING_API_KEY 与 EMBEDDING_API_BASE 是否属于同一百炼地域/工作空间。"
        ) from exc
    try:
        import torch
        print("\n--- GPU诊断信息 ---")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
        else:
            print("警告: PyTorch 未找到可用的 CUDA 设备。模型将运行在 CPU 上。")
        print("--- GPU诊断结束 ---\n")
    except ImportError:
        print("\n警告: 未安装 PyTorch。无法进行 GPU 诊断。\n")

    if rebuild and os.path.exists(PERSIST_PATH):
        print(f"--- 全量重建：正在删除旧数据库 '{PERSIST_PATH}' ---")
        shutil.rmtree(PERSIST_PATH)

    manifest = _load_manifest()

    # 1. 加载所有文档
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"错误：数据目录 '{DATA_PATH}' 不存在或为空。")
        return
    documents, processed_fingerprints, skipped_files = load_documents_from_directory(
        DATA_PATH, manifest.get("files", {})
    )
    if not documents:
        print(f"--- 没有需要更新的文件，已跳过 {skipped_files} 个未变化文件 ---")
        return
    print(f"\n成功加载 {len(documents)} 份待更新文档/数据行，跳过 {skipped_files} 个未变化文件。")

    # 2. 并行处理所有文档
    all_summaries, all_summary_metadatas, all_summary_ids = [], [], []
    all_chunks, all_chunk_metadatas, all_chunk_ids = [], [], []
    all_retrievals, all_retrieval_metadatas, all_retrieval_ids = [], [], []
    expected_by_file = Counter(doc.metadata["source_file"] for doc in documents)
    succeeded_by_file = Counter()

    # 创建进程池
    num_processes = max(1, os.cpu_count() - 1) # 留一个核心给主进程
    print(f"---" + " 使用 " + f"{num_processes}" + " 个进程并行处理文档" + " ---")
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用imap_unordered来获取进度条
        results = list(tqdm(pool.imap_unordered(process_document_worker, documents), total=len(documents), desc="摘要与切分"))

    # 3. 收集处理结果
    for result in results:
        if result:
            (doc_source, summary, summary_metadata, chunk_ids, chunk_docs, chunk_metadatas,
             retrieval_ids, retrieval_docs, retrieval_metadatas) = result
            all_summary_ids.append(doc_source)
            all_summaries.append(summary)
            all_summary_metadatas.append(summary_metadata)
            succeeded_by_file[summary_metadata["source_file"]] += 1
            all_chunk_ids.extend(chunk_ids)
            all_chunks.extend(chunk_docs)
            all_chunk_metadatas.extend(chunk_metadatas)
            all_retrieval_ids.extend(retrieval_ids)
            all_retrievals.extend(retrieval_docs)
            all_retrieval_metadatas.extend(retrieval_metadatas)

    if not all_summary_ids or not all_chunk_ids:
        print("未能成功处理任何文档，注入中止。" )
        return

    # 4. 批量存入数据库（分批次）
    print("--- 开始批量存入向量数据库 ---")
    client = chromadb.PersistentClient(path=PERSIST_PATH)
    
    # 定义一个合理的批次大小
    CHROMA_BATCH_SIZE = 4096

    # 批量存入摘要
    summary_collection = client.get_or_create_collection(SUMMARY_COLLECTION_NAME, embedding_function=embedding_function)
    total_summaries = len(all_summary_ids)
    print(f"正在分批存入 {total_summaries} 条摘要...")
    for i in tqdm(range(0, total_summaries, CHROMA_BATCH_SIZE), desc="存入摘要"):
        end_i = min(i + CHROMA_BATCH_SIZE, total_summaries)
        summary_collection.upsert(
            ids=all_summary_ids[i:end_i],
            documents=all_summaries[i:end_i],
            metadatas=all_summary_metadatas[i:end_i]
        )

    # 批量存入区块
    chunk_collection = client.get_or_create_collection(CHUNK_COLLECTION_NAME, embedding_function=embedding_function)
    total_chunks = len(all_chunk_ids)
    print(f"正在分批存入 {total_chunks} 个区块...")
    for i in tqdm(range(0, total_chunks, CHROMA_BATCH_SIZE), desc="存入区块"):
        end_i = min(i + CHROMA_BATCH_SIZE, total_chunks)
        chunk_collection.upsert(
            ids=all_chunk_ids[i:end_i],
            documents=all_chunks[i:end_i],
            metadatas=all_chunk_metadatas[i:end_i]
        )

    retrieval_collection = client.get_or_create_collection(
        RETRIEVAL_COLLECTION_NAME, embedding_function=embedding_function
    )
    print(f"正在分批存入 {len(all_retrieval_ids)} 个检索子块...")
    for i in tqdm(range(0, len(all_retrieval_ids), CHROMA_BATCH_SIZE), desc="存入检索子块"):
        end_i = min(i + CHROMA_BATCH_SIZE, len(all_retrieval_ids))
        retrieval_collection.upsert(
            ids=all_retrieval_ids[i:end_i], documents=all_retrievals[i:end_i],
            metadatas=all_retrieval_metadatas[i:end_i],
        )

    completed_files = {
        source_file
        for source_file, expected_count in expected_by_file.items()
        if succeeded_by_file[source_file] == expected_count
    }
    _delete_stale_records(
        summary_collection,
        all_summary_ids,
        all_summary_metadatas,
        completed_files,
    )
    _delete_stale_records(
        chunk_collection,
        all_chunk_ids,
        all_chunk_metadatas,
        completed_files,
    )
    _delete_stale_records(
        retrieval_collection, all_retrieval_ids,
        all_retrieval_metadatas, completed_files,
    )
    for source_file in completed_files:
        manifest["files"][source_file] = processed_fingerprints[source_file]
    manifest["pipeline_signature"] = _manifest_signature()
    _save_manifest(manifest)

    print("\n--- 增量数据注入完成 ---")
    print(f"成功更新 {len(completed_files)} 个文件，跳过 {skipped_files} 个未变化文件。")
    print(f"知识库已成功构建在 '{PERSIST_PATH}' 中。" )

# --- 辅助函数定义 ---
def load_documents_from_directory(directory_path, manifest_files=None):
    """只加载新增或发生变化的文档，并在 OCR 前跳过未变化文件。"""
    manifest_files = manifest_files or {}
    # ... (此处省略与之前版本相同的完整代码)
    documents = []
    loader_map = {
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader,
    }
    supported_files = []
    for root, _, files in os.walk(directory_path):
        lower_names = {file.lower() for file in files}
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            stem = os.path.splitext(file)[0].lower()
            # 同名 Excel 与 CSV 同时存在时优先使用 Excel，避免重复入库。
            if ext == '.csv' and (
                f"{stem}.xlsx" in lower_names or f"{stem}.xls" in lower_names
            ):
                continue
            if (
                ext in loader_map
                or ext in ['.pdf', '.xlsx', '.xls', '.csv']
                or ext in SUPPORTED_IMAGE_SUFFIXES
            ):
                supported_files.append(os.path.join(root, file))

    processed_fingerprints = {}
    skipped_files = 0
    for file_path in tqdm(supported_files, desc="加载文档"):
        source_file = os.path.normcase(os.path.abspath(file_path))
        fingerprint = _file_fingerprint(file_path)
        if manifest_files.get(source_file) == fingerprint:
            skipped_files += 1
            continue

        ext = os.path.splitext(file_path)[1].lower()
        document_start = len(documents)
        try:
            if ext == '.pdf':
                documents.extend(ocr_pdf_file(file_path))
            elif ext in SUPPORTED_IMAGE_SUFFIXES:
                documents.extend(ocr_image_file(file_path))
            elif ext in ['.xlsx', '.xls', '.csv']:
                df = pd.read_csv(file_path) if ext == '.csv' else pd.read_excel(file_path)
                for index, row in df.iterrows():
                    content = _build_tabular_text(row)
                    metadata = _tabular_metadata(file_path, row, index)
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
                documents.extend(_build_tabular_aggregates(df, file_path))
            elif ext in loader_map:
                loader = loader_map[ext](file_path, encoding='utf-8') if ext == ".txt" else loader_map[ext](file_path)
                loaded_docs = loader.load()
                # 为叙事型文档打上标签
                for doc in loaded_docs:
                    doc.metadata["data_type"] = "narrative"
                documents.extend(loaded_docs)

            new_documents = documents[document_start:]
            if new_documents:
                for doc in new_documents:
                    doc.metadata["source_file"] = source_file
                    doc.metadata["file_fingerprint"] = fingerprint
                processed_fingerprints[source_file] = fingerprint
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
    return documents, processed_fingerprints, skipped_files

if __name__ == "__main__":
    # 在Windows上使用多进程时，必须将主逻辑放在 if __name__ == '__main__': 下
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="增量构建 Agentic RAG 知识库")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="删除旧向量库并按当前模型、OCR 和分块配置全量重建",
    )
    args = parser.parse_args()
    main(rebuild=args.rebuild)
