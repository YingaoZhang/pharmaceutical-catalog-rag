#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch import large Excel package-insert tables without a web request timeout."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ENABLE_RERANK", "False")
os.environ.setdefault("PHARMA_RAG_IMPORT_MODE", "True")

import pandas as pd
import redis
from pymilvus import connections, utility

import api.main as api_main
from config import Config
from vectorization.embedding_service import EmbeddingService


embedding_service = None


def discover_excel_files(paths):
    files = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for suffix in api_main.EXCEL_SUFFIXES:
                files.extend(sorted(path.glob(f"*{suffix}")))
        elif path.is_file() and path.suffix.lower() in api_main.EXCEL_SUFFIXES:
            files.append(path)
    return files


def clear_existing_index():
    connections.connect(alias="default", host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)
    if utility.has_collection(Config.MILVUS_COLLECTION):
        utility.drop_collection(Config.MILVUS_COLLECTION)
        print(f"Dropped Milvus collection: {Config.MILVUS_COLLECTION}", flush=True)

    for path_value in (Config.LOCAL_DOC_STORE, Config.BM25_INDEX_PATH):
        path = Path(path_value)
        if path.exists():
            path.unlink()
            print(f"Deleted: {path}", flush=True)
        if path_value == Config.LOCAL_DOC_STORE:
            sibling = path.with_suffix(".json" if path.suffix == ".jsonl" else ".jsonl")
            if sibling.exists():
                sibling.unlink()
                print(f"Deleted: {sibling}", flush=True)

    clear_retrieval_cache()


def iter_row_documents(file_path, limit=None, offset=0):
    excel_file = pd.ExcelFile(file_path)
    yielded_rows = 0
    skipped_rows = 0
    for sheet_name in excel_file.sheet_names:
        frame = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
            dtype=str,
            keep_default_na=False,
        )
        frame = api_main._clean_excel_frame(frame)
        if frame.empty:
            continue

        for row_index, row in frame.iterrows():
            if skipped_rows < offset:
                skipped_rows += 1
                continue
            if limit is not None and yielded_rows >= limit:
                return

            fields = {
                str(column).strip(): api_main._normalize_excel_value(value)
                for column, value in row.items()
                if api_main._normalize_excel_value(value)
            }
            if not fields:
                continue

            yielded_rows += 1
            yield api_main._documents_from_record(
                fields=fields,
                filename=file_path.name,
                sheet_name=sheet_name,
                record_number=int(row_index) + 2,
                record_type="table_row",
            )


def flush_batch(documents, total_docs):
    if not documents:
        return 0
    if embedding_service is None:
        raise RuntimeError("Embedding service has not been initialized.")
    added = embedding_service.embed_and_store(documents)
    total_docs += added
    print(f"Indexed {added} chunks; total chunks indexed: {total_docs}", flush=True)
    return added


def build_bm25_index():
    print("BM25 index build skipped: Milvus native sparse BM25 is generated in-collection.", flush=True)
    clear_retrieval_cache()


def clear_retrieval_cache():
    try:
        client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            password=Config.REDIS_PASSWORD or None,
            db=Config.REDIS_DB,
            decode_responses=True,
        )
        keys = list(client.scan_iter(match="retrieval:*", count=100))
        if keys:
            client.delete(*keys)
        print(f"Deleted Redis retrieval cache keys: {len(keys)}", flush=True)
    except Exception as exc:
        print(f"Redis cache clear skipped: {exc}", flush=True)


def main() -> int:
    global embedding_service

    parser = argparse.ArgumentParser(
        description="Import large .xlsx/.xlsm/.xls drug instruction tables into Milvus."
    )
    parser.add_argument("paths", nargs="+", help="Excel files or directories.")
    parser.add_argument("--clear-existing", action="store_true", help="Clear Milvus/local/BM25 data before import.")
    parser.add_argument("--limit", type=int, default=None, help="Import at most N rows across all input files.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N rows across all input files.")
    parser.add_argument("--batch-docs", type=int, default=256, help="Number of chunks to encode per flush.")
    parser.add_argument("--excel-chunk-size", type=int, default=520, help="Max characters per Excel section chunk.")
    parser.add_argument(
        "--max-chunks-per-row",
        type=int,
        default=0,
        help="Max vector chunks generated per Excel row. Use 0 for no cap.",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Compatibility flag; BM25 is now generated by Milvus native sparse indexing.",
    )
    args = parser.parse_args()

    files = discover_excel_files(args.paths)
    if not files:
        print("No Excel files found.")
        return 1

    Config.EXCEL_CHUNK_SIZE = args.excel_chunk_size
    Config.EXCEL_MAX_CHUNKS_PER_ROW = args.max_chunks_per_row
    Config.EMBEDDING_BATCH_SIZE = max(1, args.batch_docs)
    embedding_service = EmbeddingService()

    if args.clear_existing:
        clear_existing_index()

    start_time = time.time()
    row_count = 0
    skipped_rows = 0
    total_chunks = 0
    pending_documents = []
    stop_import = False

    for file_path in files:
        if stop_import:
            break
        print(f"Reading {file_path}", flush=True)
        for row_documents in iter_row_documents(file_path):
            if skipped_rows < args.offset:
                skipped_rows += 1
                continue
            if args.limit is not None and row_count >= args.limit:
                stop_import = True
                break

            row_count += 1
            pending_documents.extend(row_documents)

            if len(pending_documents) >= args.batch_docs:
                total_chunks += flush_batch(pending_documents, total_chunks)
                pending_documents = []

            if row_count % 1000 == 0:
                elapsed = max(time.time() - start_time, 1)
                print(
                    f"Rows processed: {row_count}; chunks indexed: {total_chunks}; "
                    f"elapsed: {elapsed / 60:.1f} min",
                    flush=True,
                )

    if pending_documents:
        total_chunks += flush_batch(pending_documents, total_chunks)

    if total_chunks:
        print("Flushing and loading Milvus collection...", flush=True)
        embedding_service.flush_and_load()

    if not args.skip_bm25:
        build_bm25_index()

    elapsed = max(time.time() - start_time, 1)
    print(
        f"Done. Rows processed: {row_count}; chunks indexed: {total_chunks}; "
        f"elapsed: {elapsed / 60:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
