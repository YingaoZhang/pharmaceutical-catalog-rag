#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hybrid retrieval: BGE-M3 vector search in Milvus plus BM25."""

import hashlib
import json
from typing import Dict, List

import torch

from utils.torch_compat import patch_torch_metadata_version

patch_torch_metadata_version()

from config import Config
from retrieval.metadata_filter import MetadataFilter
from retrieval.reranker import Reranker
from utils.database import MilvusClient, RedisClient
from utils.logger import logger


class HybridRetriever:
    """Retrieve context with vector search and BM25 keyword matching."""

    def __init__(self):
        self.milvus_client = MilvusClient()
        from sentence_transformers import SentenceTransformer

        self.device = _resolve_device(Config.QUERY_EMBEDDING_DEVICE)
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, device=self.device)
        if Config.EMBEDDING_MAX_SEQ_LENGTH > 0:
            self.embedding_model.max_seq_length = Config.EMBEDDING_MAX_SEQ_LENGTH
        if self.device.startswith("cuda") and Config.EMBEDDING_USE_FP16:
            self.embedding_model.half()
            torch.set_float32_matmul_precision("high")
        try:
            self.redis_client = RedisClient()
        except Exception as exc:
            logger.warning(f"Redis unavailable, retrieval cache disabled: {exc}")
            self.redis_client = None

        self.collection = None
        if self.milvus_client.collection_exists(Config.MILVUS_COLLECTION):
            self.collection = self.milvus_client.get_collection(Config.MILVUS_COLLECTION)
            self.collection.load()

        self.metadata_filter = MetadataFilter()
        self.reranker = Reranker()
        logger.info("Hybrid retriever initialized with Milvus dense + native sparse BM25")

    def build_bm25_index(self, documents: List[Dict]) -> None:
        logger.info(
            "External BM25 build skipped; Milvus generates native sparse BM25 vectors from text."
        )

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        top_k = top_k or Config.FINAL_TOP_K
        cache_payload = {
            "query": query,
            "collection": Config.MILVUS_COLLECTION,
            "top_k": top_k,
            "vector_top_k": Config.VECTOR_TOP_K,
            "bm25_top_k": Config.BM25_TOP_K,
            "enable_rerank": Config.ENABLE_RERANK,
            "rerank_model": Config.RERANK_MODEL if Config.ENABLE_RERANK else "",
        }
        cache_key = "retrieval:" + hashlib.sha256(
            json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

        metadata_results, matched_terms = self.metadata_filter.filter(query)
        vector_results = self._vector_retrieve(query, Config.VECTOR_TOP_K, matched_terms=matched_terms)
        bm25_results = self._sparse_retrieve(query, Config.BM25_TOP_K, matched_terms=matched_terms)
        merged = self._merge_results(vector_results, bm25_results, metadata_results)
        coarse_results = self._rerank(merged)[: max(top_k * 3, top_k)]
        final_results = self.reranker.rerank(query, coarse_results, top_k=top_k)

        if self.redis_client:
            self.redis_client.setex(
                cache_key,
                json.dumps(final_results, ensure_ascii=False),
                Config.CACHE_EXPIRE,
            )
        return final_results

    def _vector_retrieve(self, query: str, top_k: int, matched_terms: List[str] = None) -> List[Dict]:
        if self.collection is None:
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            device=self.device,
        )[0].tolist()
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["drug_name", "text", "metadata"],
        )

        formatted = []
        for hits in results:
            for hit in hits:
                metadata_raw = hit.entity.get("metadata") or "{}"
                metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
                formatted.append(
                    {
                        "text": hit.entity.get("text") or "",
                        "drug_name": hit.entity.get("drug_name") or "",
                        "metadata": metadata or {},
                        "score": float(hit.score),
                        "vector_score": float(hit.score),
                        "bm25_score": 0.0,
                        "source": "vector",
                    }
                )
        return formatted

    def _sparse_retrieve(self, query: str, top_k: int, matched_terms: List[str] = None) -> List[Dict]:
        if self.collection is None:
            return []

        formatted = []
        try:
            results = self.collection.search(
                data=[query],
                anns_field="sparse",
                param={"metric_type": "BM25", "params": {}},
                limit=top_k,
                output_fields=["drug_name", "text", "metadata"],
            )
        except Exception as exc:
            logger.warning(f"Milvus native sparse BM25 retrieval failed: {exc}")
            return []

        for hits in results:
            for hit in hits:
                metadata_raw = hit.entity.get("metadata") or "{}"
                metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
                item = {
                    "text": hit.entity.get("text") or "",
                    "drug_name": hit.entity.get("drug_name") or "",
                    "metadata": metadata or {},
                    "score": float(hit.score),
                    "vector_score": 0.0,
                    "bm25_score": float(hit.score),
                    "source": "milvus_sparse_bm25",
                }
                if matched_terms and not self._matches_terms(item, matched_terms):
                    continue
                formatted.append(item)
        return formatted

    def _merge_results(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        metadata_results: List[Dict] = None,
    ) -> List[Dict]:
        merged: Dict[str, Dict] = {}
        metadata_results = metadata_results or []
        for rank, item in enumerate(vector_results, 1):
            key = self._dedupe_key(item)
            merged[key] = item.copy()
            merged[key].setdefault("bm25_score", 0.0)
            merged[key].setdefault("metadata_score", float(item.get("metadata_score", 0.0)))
            merged[key]["hybrid_score"] = 1.0 - rank / 100.0

        for rank, item in enumerate(bm25_results, 1):
            key = self._dedupe_key(item)
            if key not in merged:
                merged[key] = item.copy()
                merged[key].setdefault("vector_score", 0.0)
                merged[key].setdefault("metadata_score", 0.0)
                merged[key]["hybrid_score"] = 0.82 - rank / 200.0
            else:
                merged[key]["hybrid_score"] += 0.03 + max(0, Config.BM25_TOP_K - rank) * 0.001
            merged[key]["bm25_score"] = max(
                float(merged[key].get("bm25_score", 0)),
                float(item.get("bm25_score", 0)),
            )

        for rank, item in enumerate(metadata_results, 1):
            key = self._dedupe_key(item)
            if key not in merged:
                merged[key] = item.copy()
                merged[key].setdefault("vector_score", 0.0)
                merged[key].setdefault("bm25_score", 0.0)
                merged[key]["hybrid_score"] = 0.86 - rank / 250.0
            else:
                merged[key]["hybrid_score"] += 0.02 + max(0, Config.METADATA_FILTER_TOP_K - rank) * 0.0005
            merged[key]["metadata_score"] = max(
                float(merged[key].get("metadata_score", 0)),
                float(item.get("metadata_score", 0)),
            )

        return list(merged.values())

    def _rerank(self, results: List[Dict]) -> List[Dict]:
        return sorted(results, key=lambda item: item.get("hybrid_score", 0), reverse=True)

    def _matches_terms(self, doc: Dict, terms: List[str]) -> bool:
        metadata = doc.get("metadata", {}) or {}
        haystack = " ".join(
            [
                str(metadata.get("drug_name", "")),
                str(metadata.get("generic_name", "")),
                str(metadata.get("trade_name", "")),
                str(metadata.get("title", "")),
                str(metadata.get("approval_number", "")),
                str(metadata.get("manufacturer", "")),
                str(metadata.get("related_disease", "")),
                str(metadata.get("section_name", "")),
                str(metadata.get("field_name", "")),
                str(doc.get("text", "")[:500]),
            ]
        ).lower()
        return any(term in haystack for term in terms)

    def _dedupe_key(self, item: Dict) -> str:
        metadata = item.get("metadata", {}) or {}
        parts = [
            metadata.get("source_file"),
            metadata.get("sheet_name"),
            metadata.get("row_number"),
            metadata.get("section_name") or metadata.get("field_name"),
            metadata.get("chunk_index"),
        ]
        if any(part not in (None, "") for part in parts):
            return "|".join(str(part) for part in parts)
        return hashlib.sha1(item.get("text", "")[:500].encode("utf-8")).hexdigest()


def _resolve_device(device: str) -> str:
    if device and device.lower() != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
