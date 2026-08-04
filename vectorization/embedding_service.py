#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Embedding service using BGE-M3 and Milvus."""

import json
import os
from typing import Dict, List

import torch

from utils.torch_compat import patch_torch_metadata_version

patch_torch_metadata_version()

from config import Config
from utils.database import MilvusClient
from utils.logger import logger


class EmbeddingService:
    """Embed chunks with BGE-M3 and store them in Milvus."""

    def __init__(self):
        self.milvus_client = MilvusClient()
        from sentence_transformers import SentenceTransformer

        self.device = _resolve_device(Config.EMBEDDING_DEVICE)
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, device=self.device)
        if Config.EMBEDDING_MAX_SEQ_LENGTH > 0:
            self.embedding_model.max_seq_length = Config.EMBEDDING_MAX_SEQ_LENGTH
        if self.device.startswith("cuda") and Config.EMBEDDING_USE_FP16:
            self.embedding_model.half()
            torch.set_float32_matmul_precision("high")
        precision = "fp16" if self.device.startswith("cuda") and Config.EMBEDDING_USE_FP16 else "fp32"
        logger.info(f"Embedding model loaded: {Config.EMBEDDING_MODEL} on {self.device} ({precision})")

    def embed_and_store(self, chunks: List[Dict], collection_name: str = None) -> int:
        collection_name = collection_name or Config.MILVUS_COLLECTION
        clean_chunks = [chunk for chunk in chunks if (chunk.get("text") or "").strip()]
        if not clean_chunks:
            return 0

        collection = self.milvus_client.create_collection(collection_name, Config.EMBEDDING_DIM)
        batch_size = max(1, Config.EMBEDDING_BATCH_SIZE)
        flush_after_call = os.getenv("PHARMA_RAG_IMPORT_MODE", "False").lower() != "true"
        total = 0
        for start in range(0, len(clean_chunks), batch_size):
            batch = clean_chunks[start : start + batch_size]
            texts = [chunk["text"][:8192] for chunk in batch]
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=min(batch_size, Config.EMBEDDING_ENCODE_BATCH_SIZE),
                show_progress_bar=False,
                device=self.device,
            )

            drug_names = []
            languages = []
            metadata_values = []
            for chunk in batch:
                metadata = chunk.get("metadata", {}) or {}
                drug_names.append(str(metadata.get("drug_name", ""))[:512])
                languages.append(str(metadata.get("language", "zh") or "zh")[:32])
                metadata_values.append(json.dumps(metadata, ensure_ascii=False)[:4096])

            collection.insert(
                [
                    drug_names,
                    texts,
                    languages,
                    [embedding.tolist() for embedding in embeddings],
                    metadata_values,
                ]
            )
            if flush_after_call:
                collection.flush()
            self._append_local_store(batch)
            total += len(batch)

        if flush_after_call:
            collection.load()
        logger.info(f"Stored {total} chunks in Milvus collection {collection_name}")
        return total

    def embed_query(self, query: str):
        return self.embedding_model.encode([query], normalize_embeddings=True, device=self.device)[0]

    def flush_and_load(self, collection_name: str = None) -> None:
        collection_name = collection_name or Config.MILVUS_COLLECTION
        collection = self.milvus_client.get_collection(collection_name)
        collection.flush()
        collection.load()
        logger.info(f"Flushed and loaded Milvus collection {collection_name}")

    def _append_local_store(self, chunks: List[Dict]) -> None:
        os.makedirs(os.path.dirname(Config.LOCAL_DOC_STORE) or ".", exist_ok=True)
        if Config.LOCAL_DOC_STORE.lower().endswith(".jsonl"):
            with open(Config.LOCAL_DOC_STORE, "a", encoding="utf-8") as file:
                for chunk in chunks:
                    file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            return

        existing = []
        if os.path.exists(Config.LOCAL_DOC_STORE):
            try:
                with open(Config.LOCAL_DOC_STORE, "r", encoding="utf-8") as file:
                    existing = json.load(file)
            except Exception:
                existing = []
        existing.extend(chunks)
        with open(Config.LOCAL_DOC_STORE, "w", encoding="utf-8") as file:
            json.dump(existing, file, ensure_ascii=False, indent=2)


def _resolve_device(device: str) -> str:
    if device and device.lower() != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
