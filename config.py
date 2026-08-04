#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Application configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _resolve_local_hf_model(model_name: str) -> str:
    """Prefer an existing local Hugging Face snapshot over a remote model id."""
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path.resolve())

    cache_roots = [
        _project_path(os.getenv("HF_HOME", str(BASE_DIR / "models" / "hf"))),
        _project_path(os.getenv("SENTENCE_TRANSFORMERS_HOME", str(BASE_DIR / "models" / "hf"))),
        BASE_DIR / "models" / "hf",
        BASE_DIR / "models" / "hf" / "hub",
    ]
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    for root in cache_roots:
        snapshots_dir = root / model_dir_name / "snapshots"
        if not snapshots_dir.exists():
            continue
        snapshots = sorted(
            [path for path in snapshots_dir.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0].resolve())
    return model_name


def _project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


class Config:
    """Centralized settings loaded from environment variables."""

    # Ollama generation settings.
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # Embedding, hybrid retrieval, metadata filtering, and reranking.
    EMBEDDING_MODEL = _resolve_local_hf_model(os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")
    QUERY_EMBEDDING_DEVICE = os.getenv("QUERY_EMBEDDING_DEVICE", "cpu")
    EMBEDDING_USE_FP16 = os.getenv("EMBEDDING_USE_FP16", "True").lower() == "true"
    EMBEDDING_ENCODE_BATCH_SIZE = int(os.getenv("EMBEDDING_ENCODE_BATCH_SIZE", "64"))
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
    BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
    BM25_B = float(os.getenv("BM25_B", "0.75"))
    VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "40"))
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", "40"))
    FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "8"))
    HYBRID_WEIGHT = float(os.getenv("HYBRID_WEIGHT", "0.6"))
    METADATA_FILTER_TOP_K = int(os.getenv("METADATA_FILTER_TOP_K", "80"))
    ENABLE_METADATA_FILTER = os.getenv("ENABLE_METADATA_FILTER", "True").lower() == "true"
    ENABLE_RERANK = os.getenv("ENABLE_RERANK", "True").lower() == "true"
    RERANK_MODEL = _resolve_local_hf_model(os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"))
    RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")
    RERANK_USE_FP16 = os.getenv("RERANK_USE_FP16", "True").lower() == "true"
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "8"))
    RERANK_WEIGHT = float(os.getenv("RERANK_WEIGHT", "0.35"))
    MIN_SUPPORTED_SOURCES = int(os.getenv("MIN_SUPPORTED_SOURCES", "1"))

    # Milvus and Redis.
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "pharmaceutical_docs")
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    CACHE_EXPIRE = int(os.getenv("CACHE_EXPIRE", "3600"))

    # Data and upload paths.
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    EXCEL_CHUNK_SIZE = int(os.getenv("EXCEL_CHUNK_SIZE", "520"))
    EXCEL_CHUNK_OVERLAP = int(os.getenv("EXCEL_CHUNK_OVERLAP", "80"))
    EXCEL_MAX_CHUNKS_PER_ROW = int(os.getenv("EXCEL_MAX_CHUNKS_PER_ROW", "0"))
    EXCEL_MAX_UPLOAD_ROWS = int(os.getenv("EXCEL_MAX_UPLOAD_ROWS", "5000"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    EMBEDDING_MAX_SEQ_LENGTH = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "512"))
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "100"))  # MB
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    LOCAL_DOC_STORE = os.getenv("LOCAL_DOC_STORE", "./data/documents.json")
    BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")

    # App/runtime settings.
    API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    ALLOW_EXTERNAL_ACCESS = os.getenv("ALLOW_EXTERNAL_ACCESS", "False").lower() == "true"
    ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "False").lower() == "true"
