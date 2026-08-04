#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-encoder reranking for retrieved chunks."""

from typing import Dict, List

import torch

from config import Config
from utils.logger import logger
from utils.torch_compat import patch_torch_metadata_version


class Reranker:
    """Rerank candidate chunks with a BGE reranker model."""

    def __init__(self):
        self.model = None
        if not Config.ENABLE_RERANK:
            return
        try:
            patch_torch_metadata_version()
            from sentence_transformers import CrossEncoder

            self.device = _resolve_device(Config.RERANK_DEVICE)
            self.model = CrossEncoder(Config.RERANK_MODEL, device=self.device)
            if self.device.startswith("cuda") and Config.RERANK_USE_FP16:
                self.model.model.half()
                torch.set_float32_matmul_precision("high")
            precision = "fp16" if self.device.startswith("cuda") and Config.RERANK_USE_FP16 else "fp32"
            logger.info(f"Reranker loaded: {Config.RERANK_MODEL} on {self.device} ({precision})")
        except Exception as exc:
            logger.warning(f"Reranker unavailable, falling back to hybrid score: {exc}")
            self.model = None

    def rerank(self, query: str, candidates: List[Dict], top_k: int = None) -> List[Dict]:
        top_k = top_k or Config.RERANK_TOP_K
        if not candidates:
            return []
        if self.model is None:
            return sorted(candidates, key=lambda item: item.get("hybrid_score", 0), reverse=True)[:top_k]

        pairs = [[query, item.get("text", "")[:1800]] for item in candidates]
        scores = self.model.predict(pairs)
        max_score = max(float(score) for score in scores) or 1.0
        min_score = min(float(score) for score in scores)
        span = max(max_score - min_score, 1e-6)

        reranked = []
        for item, score in zip(candidates, scores):
            normalized = (float(score) - min_score) / span
            combined = (
                (1 - Config.RERANK_WEIGHT) * float(item.get("hybrid_score", 0))
                + Config.RERANK_WEIGHT * normalized
            )
            enriched = item.copy()
            enriched["rerank_score"] = float(score)
            enriched["final_score"] = combined
            reranked.append(enriched)

        return sorted(reranked, key=lambda item: item.get("final_score", 0), reverse=True)[:top_k]


def _resolve_device(device: str) -> str:
    if device and device.lower() != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
