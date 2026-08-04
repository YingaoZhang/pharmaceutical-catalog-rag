#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI service for BGE-M3 + Milvus + BM25 + Ollama RAG."""

import io
import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import Config
from core.qa_pipeline import QAPipeline
from retrieval.hybrid_retriever import HybridRetriever
from utils.logger import logger
from vectorization.embedding_service import EmbeddingService

app = FastAPI(
    title="Pharmaceutical Catalog RAG",
    description="BGE-M3 + Milvus + BM25 retrieval with Ollama generation",
    version="1.1.0-rag-ollama",
)

allowed_origins = ["*"] if Config.ALLOW_EXTERNAL_ACCESS else [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.getenv("PHARMA_RAG_IMPORT_MODE", "False").lower() == "true":
    qa_pipeline = None
    embedding_service = None
else:
    qa_pipeline = QAPipeline()
    embedding_service = EmbeddingService()


class QARequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None


class QAResponse(BaseModel):
    answer: str
    sources: List[Dict]
    source_type: str
    confidence: float
    response_time: float
    verification: Optional[Dict[str, Any]] = None


class DocumentIn(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class VectorizeRequest(BaseModel):
    documents: List[DocumentIn]


EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")
DRUG_NAME_ALIASES = (
    "drug_name",
    "drug",
    "name",
    "药品名称",
    "药品名",
    "通用名称",
    "商品名称",
    "产品名称",
    "中文名称",
    "名称",
)
TITLE_ALIASES = (
    "title",
    "标题",
    "说明书标题",
    "药品名称",
    "通用名称",
    "商品名称",
)
SECTION_ALIASES = (
    "成份",
    "性状",
    "适应症",
    "功能主治",
    "规格",
    "用法用量",
    "不良反应",
    "禁忌",
    "注意事项",
    "孕妇及哺乳期妇女用药",
    "儿童用药",
    "老年用药",
    "药物相互作用",
    "药理毒理",
    "药代动力学",
    "贮藏",
    "包装",
    "有效期",
    "批准文号",
    "生产企业",
)


# Column aliases for instruction workbooks where each row is one drug and
# each column is a package-insert field.
DRUG_NAME_ALIASES = (
    "drug_name",
    "drug",
    "name",
    "\u901a\u7528\u540d\u79f0",
    "\u5546\u54c1\u540d\u79f0",
    "\u836f\u54c1\u540d\u79f0",
    "\u836f\u54c1\u540d",
    "\u6807\u9898",
    "\u4ea7\u54c1\u540d\u79f0",
    "\u4e2d\u6587\u540d\u79f0",
    "\u540d\u79f0",
)
TITLE_ALIASES = (
    "title",
    "\u6807\u9898",
    "\u8bf4\u660e\u4e66\u6807\u9898",
    "\u836f\u54c1\u540d\u79f0",
    "\u901a\u7528\u540d\u79f0",
    "\u5546\u54c1\u540d\u79f0",
)
GENERIC_NAME_ALIASES = ("generic_name", "\u901a\u7528\u540d\u79f0")
TRADE_NAME_ALIASES = ("trade_name", "\u5546\u54c1\u540d\u79f0")
SOURCE_URL_ALIASES = ("source_url", "url", "\u6807\u9898\u94fe\u63a5", "\u94fe\u63a5")
APPROVAL_NUMBER_ALIASES = ("approval_number", "\u6279\u51c6\u6587\u53f7")
MANUFACTURER_ALIASES = ("manufacturer", "\u751f\u4ea7\u4f01\u4e1a")
DRUG_CATEGORY_ALIASES = ("drug_category", "\u836f\u54c1\u5206\u7c7b")
DRUG_PROPERTY_ALIASES = ("drug_property", "\u836f\u54c1\u6027\u8d28")
RELATED_DISEASE_ALIASES = ("related_disease", "\u76f8\u5173\u75be\u75c5")
FIELD_ORDER = (
    "\u6807\u9898",
    "\u901a\u7528\u540d\u79f0",
    "\u5546\u54c1\u540d\u79f0",
    "\u6c49\u8bed\u62fc\u97f3",
    "\u6279\u51c6\u6587\u53f7",
    "\u836f\u54c1\u5206\u7c7b",
    "\u751f\u4ea7\u4f01\u4e1a",
    "\u836f\u54c1\u6027\u8d28",
    "\u76f8\u5173\u75be\u75c5",
    "\u6027\u72b6",
    "\u4e3b\u8981\u6210\u4efd",
    "\u9002\u5e94\u75c7",
    "\u89c4\u683c",
    "\u4e0d\u826f\u53cd\u5e94",
    "\u7528\u6cd5\u7528\u91cf",
    "\u7981\u5fcc",
    "\u6ce8\u610f\u4e8b\u9879",
    "\u5b55\u5987\u53ca\u54fa\u4e73\u671f\u5987\u5973\u7528\u836f",
    "\u513f\u7ae5\u7528\u836f",
    "\u8001\u4eba\u7528\u836f",
    "\u8001\u5e74\u7528\u836f",
    "\u836f\u7269\u76f8\u4e92\u4f5c\u7528",
    "\u836f\u7406\u6bd2\u7406",
    "\u836f\u4ee3\u52a8\u529b\u5b66",
    "\u8d2e\u85cf",
    "\u5305\u88c5",
    "\u6709\u6548\u671f",
)
SECTION_ALIASES = (
    "\u6210\u4efd",
    "\u4e3b\u8981\u6210\u4efd",
    "\u6027\u72b6",
    "\u9002\u5e94\u75c7",
    "\u529f\u80fd\u4e3b\u6cbb",
    "\u89c4\u683c",
    "\u7528\u6cd5\u7528\u91cf",
    "\u4e0d\u826f\u53cd\u5e94",
    "\u7981\u5fcc",
    "\u6ce8\u610f\u4e8b\u9879",
    "\u5b55\u5987\u53ca\u54fa\u4e73\u671f\u5987\u5973\u7528\u836f",
    "\u513f\u7ae5\u7528\u836f",
    "\u8001\u4eba\u7528\u836f",
    "\u8001\u5e74\u7528\u836f",
    "\u836f\u7269\u76f8\u4e92\u4f5c\u7528",
    "\u836f\u7406\u6bd2\u7406",
    "\u836f\u4ee3\u52a8\u529b\u5b66",
    "\u8d2e\u85cf",
    "\u5305\u88c5",
    "\u6709\u6548\u671f",
    "\u6279\u51c6\u6587\u53f7",
    "\u751f\u4ea7\u4f01\u4e1a",
)

STRUCTURED_EXCEL_RECORD_TYPES = {"table_row", "key_value_sheet"}
EXCEL_SOURCE_FIELD_ALIASES = (
    "source_url",
    "url",
    "\u6807\u9898\u94fe\u63a5",
    "\u94fe\u63a5",
    "\u7f16\u53f7",
)
EXCEL_FIELD_GROUPS = (
    ("\u9002\u5e94\u75c7", ("\u9002\u5e94\u75c7", "\u529f\u80fd\u4e3b\u6cbb", "r3")),
    ("\u7528\u6cd5\u7528\u91cf", ("\u7528\u6cd5\u7528\u91cf",)),
    ("\u4e0d\u826f\u53cd\u5e94", ("\u4e0d\u826f\u53cd\u5e94",)),
    ("\u7981\u5fcc", ("\u7981\u5fcc",)),
    ("\u6ce8\u610f\u4e8b\u9879", ("\u6ce8\u610f\u4e8b\u9879",)),
    (
        "\u7279\u6b8a\u4eba\u7fa4\u7528\u836f",
        (
            "\u5b55\u5987\u53ca\u54fa\u4e73\u671f\u5987\u5973\u7528\u836f",
            "\u513f\u7ae5\u7528\u836f",
            "\u8001\u4eba\u7528\u836f",
            "\u8001\u5e74\u7528\u836f",
        ),
    ),
    ("\u836f\u7269\u76f8\u4e92\u4f5c\u7528", ("\u836f\u7269\u76f8\u4e92\u4f5c\u7528",)),
    ("\u836f\u7406\u6bd2\u7406", ("\u836f\u7406\u6bd2\u7406",)),
    ("\u836f\u4ee3\u52a8\u529b\u5b66", ("\u836f\u4ee3\u52a8\u529b\u5b66",)),
    ("\u8d2e\u85cf", ("\u8d2e\u85cf",)),
    ("\u6709\u6548\u671f", ("\u6709\u6548\u671f",)),
)


@app.get("/")
async def root():
    return {
        "message": "Pharmaceutical Catalog RAG",
        "version": "1.1.0-rag-ollama",
        "retrieval": "BGE-M3 + Milvus + BM25",
        "rerank": Config.RERANK_MODEL if Config.ENABLE_RERANK else "disabled",
        "llm": Config.OLLAMA_MODEL,
        "ollama_base_url": Config.OLLAMA_BASE_URL,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "collection": Config.MILVUS_COLLECTION,
        "llm": Config.OLLAMA_MODEL,
        "rerank": Config.RERANK_MODEL if Config.ENABLE_RERANK else "disabled",
        "bm25_index": "milvus_native_sparse",
        "local_store": os.path.exists(Config.LOCAL_DOC_STORE),
    }


@app.post("/api/v1/qa/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")
    result = qa_pipeline.answer(question=question, user_context=request.context)
    return QAResponse(**result)


@app.post("/api/v1/documents/vectorize")
async def vectorize_documents(request: VectorizeRequest):
    documents = [
        {"text": item.text, "metadata": item.metadata or {}}
        for item in request.documents
        if item.text.strip()
    ]
    added = _index_documents(documents)
    return {"status": "success", "added": added}


@app.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename or "uploaded"
        if len(content) > Config.MAX_UPLOAD_SIZE * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds MAX_UPLOAD_SIZE")

        filename_lower = filename.lower()
        records_count = None
        if filename_lower.endswith(".pdf"):
            documents = _documents_from_pdf(content, filename)
        elif filename_lower.endswith(EXCEL_SUFFIXES):
            documents = _documents_from_excel(content, filename)
            records_count = _count_excel_records(documents)
        else:
            text = _decode_upload(content)
            chunks = _chunk_text(text, Config.CHUNK_SIZE)
            documents = [
                {
                    "text": chunk,
                    "metadata": {
                        "source_file": filename,
                        "source": filename,
                        "section": f"chunk-{index}",
                    },
                }
                for index, chunk in enumerate(chunks, 1)
            ]
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No readable text was found. If this is a scanned PDF, OCR is required.",
            )
        added = _index_documents(documents)

        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        with open(os.path.join(Config.UPLOAD_DIR, os.path.basename(filename)), "wb") as target:
            target.write(content)

        response_payload = {
            "status": "success",
            "filename": filename,
            "chunks_count": added,
            "message": "Document indexed into Milvus with native sparse BM25",
        }
        if records_count is not None:
            response_payload["records_count"] = records_count
        return response_payload
    except Exception as exc:
        logger.error(f"Document upload failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/stats")
async def get_stats():
    local_store_exists = os.path.exists(Config.LOCAL_DOC_STORE)
    collection = getattr(getattr(qa_pipeline, "retriever", None), "collection", None)
    try:
        indexed_chunks = int(collection.num_entities) if collection is not None else 0
    except Exception as exc:
        logger.warning(f"Failed to read Milvus entity count: {exc}")
        indexed_chunks = 0
    return {
        "system_status": "running",
        "collection": Config.MILVUS_COLLECTION,
        "retrieval": "BGE-M3 dense + Milvus native sparse BM25",
        "rerank": Config.RERANK_MODEL if Config.ENABLE_RERANK else "disabled",
        "rerank_enabled": Config.ENABLE_RERANK,
        "rerank_device": Config.RERANK_DEVICE,
        "llm": Config.OLLAMA_MODEL,
        "ollama_base_url": Config.OLLAMA_BASE_URL,
        "embedding_device": Config.EMBEDDING_DEVICE,
        "query_embedding_device": Config.QUERY_EMBEDDING_DEVICE,
        "documents": indexed_chunks,
        "local_store": os.path.abspath(Config.LOCAL_DOC_STORE),
        "local_store_exists": local_store_exists,
        "bm25_index": "Milvus native sparse BM25",
        "bm25_index_exists": True,
    }


def _index_documents(documents: List[Dict]) -> int:
    if not documents:
        return 0
    seen_keys = {_document_dedupe_key(document) for document in _load_local_documents()}
    new_documents = []
    for document in documents:
        key = _document_dedupe_key(document)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        new_documents.append(document)
    if not new_documents:
        return 0

    added = embedding_service.embed_and_store(new_documents)
    all_documents = _load_local_documents()
    qa_pipeline.retriever.build_bm25_index(all_documents)
    qa_pipeline.retriever = HybridRetriever()
    _clear_retrieval_cache()
    return added


def _load_local_documents() -> List[Dict]:
    if not os.path.exists(Config.LOCAL_DOC_STORE):
        return []
    with open(Config.LOCAL_DOC_STORE, "r", encoding="utf-8") as file:
        first_char = file.read(1)
        file.seek(0)
        if first_char == "[" or first_char == "{":
            try:
                payload = json.load(file)
                return payload if isinstance(payload, list) else payload.get("documents", [])
            except json.JSONDecodeError:
                file.seek(0)

        documents = []
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(f"Skipping invalid local document line: {exc}")
        return documents


def _document_dedupe_key(document: Dict) -> str:
    metadata = document.get("metadata", {}) or {}
    payload = {
        "source": metadata.get("source") or metadata.get("source_file") or "",
        "section": metadata.get("section") or "",
        "drug_name": metadata.get("drug_name") or "",
        "text": document.get("text") or "",
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _clear_retrieval_cache() -> None:
    if qa_pipeline is None:
        return
    redis_client = qa_pipeline.retriever.redis_client
    if not redis_client:
        return
    try:
        keys = list(redis_client.client.scan_iter(match="retrieval:*", count=100))
        if keys:
            redis_client.client.delete(*keys)
            logger.info(f"Cleared {len(keys)} retrieval cache entries")
    except Exception as exc:
        logger.warning(f"Failed to clear retrieval cache: {exc}")


def _decode_upload(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _documents_from_excel(content: bytes, filename: str) -> List[Dict]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Excel import requires pandas and openpyxl. Please install project dependencies.",
        ) from exc

    try:
        excel_file = pd.ExcelFile(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read Excel file: {exc}") from exc

    documents: List[Dict] = []
    for sheet_name in excel_file.sheet_names:
        try:
            frame = pd.read_excel(
                excel_file,
                sheet_name=sheet_name,
                dtype=str,
                keep_default_na=False,
            )
        except Exception as exc:
            logger.warning(f"Failed to read sheet {sheet_name} from {filename}: {exc}")
            continue

        frame = _clean_excel_frame(frame)
        if frame.empty:
            continue
        if len(frame) > Config.EXCEL_MAX_UPLOAD_ROWS:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Excel sheet '{sheet_name}' has {len(frame)} rows. "
                    f"Web upload is limited to {Config.EXCEL_MAX_UPLOAD_ROWS} rows; "
                    "use scripts/import_excel_large.py for batch import."
                ),
            )

        if _looks_like_key_value_sheet(frame):
            documents.extend(_documents_from_key_value_sheet(frame, filename, sheet_name))
        else:
            documents.extend(_documents_from_table_sheet(frame, filename, sheet_name))

    return documents


def _clean_excel_frame(frame):
    frame = frame.fillna("")
    frame.columns = [
        _normalize_excel_value(column) or f"column_{index + 1}"
        for index, column in enumerate(frame.columns)
    ]
    non_empty_columns = [
        column
        for column in frame.columns
        if any(_normalize_excel_value(value) for value in frame[column].tolist())
    ]
    frame = frame[non_empty_columns]
    if frame.empty:
        return frame
    non_empty_rows = frame.apply(
        lambda row: any(_normalize_excel_value(value) for value in row.tolist()),
        axis=1,
    )
    return frame.loc[non_empty_rows].reset_index(drop=True)


def _looks_like_key_value_sheet(frame) -> bool:
    if len(frame.columns) > 4 or len(frame) < 2:
        return False
    first_column_values = [
        _normalize_column_name(value)
        for value in frame.iloc[:, 0].tolist()
        if _normalize_excel_value(value)
    ]
    aliases = [_normalize_column_name(alias) for alias in DRUG_NAME_ALIASES + SECTION_ALIASES]
    hits = sum(
        1
        for value in first_column_values
        if any(alias and alias in value for alias in aliases)
    )
    return hits >= 2


def _documents_from_key_value_sheet(frame, filename: str, sheet_name: str) -> List[Dict]:
    fields: Dict[str, str] = {}
    for _, row in frame.iterrows():
        key = _normalize_excel_value(row.iloc[0])
        values = [
            _normalize_excel_value(value)
            for value in row.iloc[1:].tolist()
            if _normalize_excel_value(value)
        ]
        if key and values:
            fields[key] = "\n".join(values)

    return _documents_from_record(
        fields=fields,
        filename=filename,
        sheet_name=sheet_name,
        record_number=1,
        record_type="key_value_sheet",
    )


def _documents_from_table_sheet(frame, filename: str, sheet_name: str) -> List[Dict]:
    documents: List[Dict] = []
    for row_index, row in frame.iterrows():
        fields = {
            str(column).strip(): _normalize_excel_value(value)
            for column, value in row.items()
            if _normalize_excel_value(value)
        }
        documents.extend(
            _documents_from_record(
                fields=fields,
                filename=filename,
                sheet_name=sheet_name,
                record_number=int(row_index) + 2,
                record_type="table_row",
            )
        )
    return documents


def _documents_from_record(
    fields: Dict[str, str],
    filename: str,
    sheet_name: str,
    record_number: int,
    record_type: str,
) -> List[Dict]:
    if not fields:
        return []

    generic_name = _find_alias_value(fields, GENERIC_NAME_ALIASES)
    trade_name = _find_alias_value(fields, TRADE_NAME_ALIASES)
    drug_name = generic_name or _find_alias_value(fields, DRUG_NAME_ALIASES) or _filename_stem(filename)
    title = _find_alias_value(fields, TITLE_ALIASES) or drug_name
    base_metadata = {
        "source_file": filename,
        "source": f"{filename}/{sheet_name}",
        "sheet_name": sheet_name,
        "row_number": record_number,
        "record_type": record_type,
        "drug_name": drug_name,
        "generic_name": generic_name,
        "trade_name": trade_name,
        "title": title,
        "source_url": _find_alias_value(fields, SOURCE_URL_ALIASES),
        "approval_number": _find_alias_value(fields, APPROVAL_NUMBER_ALIASES),
        "manufacturer": _find_alias_value(fields, MANUFACTURER_ALIASES),
        "drug_category": _find_alias_value(fields, DRUG_CATEGORY_ALIASES),
        "drug_property": _find_alias_value(fields, DRUG_PROPERTY_ALIASES),
        "related_disease": _find_alias_value(fields, RELATED_DISEASE_ALIASES),
        "language": "zh",
    }

    if record_type in STRUCTURED_EXCEL_RECORD_TYPES:
        return _documents_from_structured_excel_record(fields, base_metadata)

    chunks = _chunk_text(_format_excel_record_text(fields, drug_name), Config.CHUNK_SIZE)
    return [
        {
            "text": chunk,
            "metadata": {
                **base_metadata,
                "section": f"{sheet_name}-row-{record_number}-chunk-{index}",
                "chunk_index": index,
            },
        }
        for index, chunk in enumerate(chunks, 1)
    ]


def _documents_from_structured_excel_record(fields: Dict[str, str], base_metadata: Dict) -> List[Dict]:
    documents: List[Dict] = []
    for group_index, group in enumerate(_group_excel_fields(fields), 1):
        section_name = group["section_name"]
        field_names = group["field_names"]
        body = _format_group_body(group["items"])
        header = _format_excel_chunk_header(base_metadata, section_name)
        chunks = _chunk_labeled_text(
            header=header,
            body=body,
            max_len=Config.EXCEL_CHUNK_SIZE,
            overlap=Config.EXCEL_CHUNK_OVERLAP,
        )
        total_parts = len(chunks)
        for part_index, chunk in enumerate(chunks, 1):
            documents.append(
                {
                    "text": chunk,
                    "metadata": {
                        **base_metadata,
                        "section": (
                            f"{base_metadata['sheet_name']}-row-{base_metadata['row_number']}-"
                            f"{section_name}-part-{part_index}"
                        ),
                        "section_name": section_name,
                        "field_name": "\u3001".join(field_names),
                        "field_group_index": group_index,
                        "chunk_index": part_index,
                        "chunk_total": total_parts,
                    },
                }
            )

    max_chunks = Config.EXCEL_MAX_CHUNKS_PER_ROW
    if max_chunks > 0 and len(documents) > max_chunks:
        kept = documents[:max_chunks]
        kept[-1]["text"] = (
            f"{kept[-1]['text']}\n"
            f"\u5907\u6ce8\uff1a\u8be5\u8bf4\u660e\u4e66\u5df2\u6309\u680f\u76ee\u5207\u5206\uff0c"
            f"\u4ec5\u4fdd\u7559\u524d {max_chunks} \u4e2a\u7247\u6bb5\uff0c"
            f"\u53e6\u6709 {len(documents) - max_chunks} \u4e2a\u7247\u6bb5\u672a\u7f16\u7801\u3002"
        )
        return kept
    return documents


def _group_excel_fields(fields: Dict[str, str]) -> List[Dict]:
    groups = []
    used_keys = set()
    for section_name, aliases in EXCEL_FIELD_GROUPS:
        items = []
        for key, value in fields.items():
            if key in used_keys or not value:
                continue
            if _field_matches_aliases(key, aliases):
                items.append((key, value))
                used_keys.add(key)
        if items:
            groups.append(
                {
                    "section_name": section_name,
                    "field_names": [key for key, _ in items],
                    "items": items,
                }
            )

    return groups


def _field_matches_aliases(key: str, aliases) -> bool:
    normalized_key = _normalize_column_name(key)
    normalized_aliases = [_normalize_column_name(alias) for alias in aliases]
    return any(
        normalized_key == alias
        or normalized_key.endswith(alias)
        or alias.endswith(normalized_key)
        for alias in normalized_aliases
        if alias
    )


def _format_group_body(items: List[tuple]) -> str:
    return "\n".join(f"{key}\uff1a{value}" for key, value in items if value)


def _format_excel_chunk_header(metadata: Dict, section_name: str) -> str:
    lines = []
    for label, key in (
        ("\u836f\u54c1\u540d\u79f0", "drug_name"),
    ):
        value = metadata.get(key, "")
        if value:
            lines.append(f"{label}\uff1a{value}")
    lines.append(f"\u680f\u76ee\uff1a{section_name}")
    return "\n".join(lines)


def _chunk_labeled_text(header: str, body: str, max_len: int, overlap: int) -> List[str]:
    max_len = max(max_len, 300)
    content_prefix = "\u5185\u5bb9\uff1a"
    body_budget = max(max_len - len(header) - len(content_prefix) - 1, 200)
    parts = _split_text_with_overlap(body, body_budget, overlap)
    return [f"{header}\n{content_prefix}{part}".strip() for part in parts] or [header]


def _split_text_with_overlap(text: str, max_len: int, overlap: int) -> List[str]:
    cleaned = "\n".join(line.strip() for line in str(text).splitlines() if line.strip())
    if not cleaned:
        return []
    if len(cleaned) <= max_len:
        return [cleaned]

    parts = []
    start = 0
    punctuation = "\n\u3002\uff1b;,\uff0c.!?\uff01\uff1f\u3001"
    while start < len(cleaned):
        end = min(start + max_len, len(cleaned))
        if end < len(cleaned):
            split_at = max(cleaned.rfind(mark, start, end) for mark in punctuation)
            if split_at > start + max_len // 2:
                end = split_at + 1
        part = cleaned[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(cleaned):
            break
        next_start = max(end - max(0, overlap), start + 1)
        start = next_start
    return parts


def _count_excel_records(documents: List[Dict]) -> int:
    records = set()
    for document in documents:
        metadata = document.get("metadata", {}) or {}
        records.add(
            (
                metadata.get("source_file", ""),
                metadata.get("sheet_name", ""),
                metadata.get("row_number", ""),
            )
        )
    return len(records)


def _format_excel_record_text(fields: Dict[str, str], drug_name: str) -> str:
    lines = []
    if drug_name and not _find_alias_value(fields, DRUG_NAME_ALIASES):
        lines.append(f"药品名称：{drug_name}")
    for key, value in fields.items():
        if value:
            lines.append(f"{key}：{value}")
    return "\n".join(lines)


def _find_alias_value(fields: Dict[str, str], aliases) -> str:
    normalized_aliases = [_normalize_column_name(alias) for alias in aliases]
    for key, value in fields.items():
        normalized_key = _normalize_column_name(key)
        if any(
            normalized_key == alias
            or normalized_key.endswith(alias)
            or alias.endswith(normalized_key)
            for alias in normalized_aliases
            if alias
        ):
            return value
    return ""


def _normalize_excel_value(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _normalize_column_name(value) -> str:
    text = _normalize_excel_value(value).lower()
    return re.sub(r"[\s:_：\-—（）()\[\]【】/\\]+", "", text)


def _filename_stem(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0]


def _format_excel_record_text(fields: Dict[str, str], drug_name: str) -> str:
    lines = []
    if drug_name and not _find_alias_value(fields, DRUG_NAME_ALIASES):
        lines.append(f"\u836f\u54c1\u540d\u79f0\uff1a{drug_name}")

    ordered_fields = []
    used_keys = set()
    for field_name in FIELD_ORDER:
        original_key = _find_alias_key(fields, (field_name,))
        if original_key:
            ordered_fields.append((original_key, fields[original_key]))
            used_keys.add(original_key)

    for key, value in fields.items():
        if key not in used_keys:
            ordered_fields.append((key, value))

    for key, value in ordered_fields:
        if value:
            lines.extend(_format_field_lines(key, value))
    return "\n".join(lines)


def _format_field_lines(key: str, value: str) -> List[str]:
    prefix = f"{key}\uff1a"
    max_value_len = max(Config.CHUNK_SIZE - len(prefix), 80)
    value_parts = _split_long_text(value, max_value_len)
    if not value_parts:
        return []
    return [prefix + value_parts[0]] + [prefix + part for part in value_parts[1:]]


def _find_alias_value(fields: Dict[str, str], aliases) -> str:
    key = _find_alias_key(fields, aliases)
    return fields.get(key, "") if key else ""


def _find_alias_key(fields: Dict[str, str], aliases) -> str:
    normalized_aliases = [_normalize_column_name(alias) for alias in aliases]
    for key in fields:
        normalized_key = _normalize_column_name(key)
        if any(
            normalized_key == alias
            or normalized_key.endswith(alias)
            or alias.endswith(normalized_key)
            for alias in normalized_aliases
            if alias
        ):
            return key
    return ""


def _documents_from_pdf(content: bytes, filename: str) -> List[Dict]:
    pages = _extract_pdf_pages(content)
    documents = []
    for page_number, page_text in pages:
        chunks = _chunk_text(page_text, Config.CHUNK_SIZE)
        for index, chunk in enumerate(chunks, 1):
            documents.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source_file": filename,
                        "source": filename,
                        "page": page_number,
                        "section": f"page-{page_number}-chunk-{index}",
                        "language": "zh",
                    },
                }
            )
    return documents


def _extract_pdf_pages(content: bytes) -> List[tuple]:
    try:
        import pdfplumber

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            pages = []
            with pdfplumber.open(tmp_path) as pdf:
                for page_index, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append((page_index, text))
            if pages:
                return pages
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as exc:
        logger.warning(f"pdfplumber extraction failed: {exc}")

    try:
        import io
        from PyPDF2 import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page_index, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((page_index, text))
        return pages
    except Exception as exc:
        logger.warning(f"PyPDF2 extraction failed: {exc}")
        return []


def _chunk_text(text: str, chunk_size: int) -> List[str]:
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    chunks = []
    current = []
    current_len = 0
    for paragraph in cleaned.split("\n"):
        if len(paragraph) > chunk_size:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            chunks.extend(_split_long_text(paragraph, chunk_size))
            continue
        if current and current_len + len(paragraph) > chunk_size:
            chunks.append("\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += len(paragraph)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _chunk_excel_record_text(text: str, chunk_size: int, max_chunks: int) -> List[str]:
    chunks = _chunk_text(text, chunk_size)
    if max_chunks <= 0 or len(chunks) <= max_chunks:
        return chunks

    kept = chunks[:max_chunks]
    dropped_count = len(chunks) - max_chunks
    kept[-1] = (
        f"{kept[-1]}\n"
        f"\u5907\u6ce8\uff1a\u8be5\u8bf4\u660e\u4e66\u5185\u5bb9\u8f83\u957f\uff0c"
        f"\u5df2\u622a\u53d6\u524d {max_chunks} \u4e2a\u7247\u6bb5\u5165\u5e93\uff0c"
        f"\u53e6\u6709 {dropped_count} \u4e2a\u7247\u6bb5\u672a\u7f16\u7801\u3002"
    )
    return kept


def _split_long_text(text: str, max_len: int) -> List[str]:
    text = _normalize_excel_value(text)
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    parts = []
    start = 0
    punctuation = "\u3002\uff1b\uff1b\uff0c\uff0c.!?;,"
    while start < len(text):
        end = min(start + max_len, len(text))
        if end < len(text):
            split_at = max(text.rfind(mark, start, end) for mark in punctuation)
            if split_at > start + max_len // 2:
                end = split_at + 1
        parts.append(text[start:end].strip())
        start = end
    return [part for part in parts if part]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=Config.DEBUG)
