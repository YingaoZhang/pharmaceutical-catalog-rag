#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer generation through local Ollama."""

import time
from typing import Dict, List

import requests

from config import Config
from generation.answer_verifier import AnswerVerifier
from utils.logger import logger


class AnswerGenerator:
    """Generate grounded answers with Ollama while keeping full RAG retrieval."""

    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL.rstrip("/")
        self.model = Config.OLLAMA_MODEL
        self.verifier = AnswerVerifier()
        logger.info(f"Answer generator initialized with Ollama model: {self.model}")

    def generate(self, question: str, context: List[Dict]) -> Dict:
        start_time = time.time()
        sources = self._extract_sources(context)
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是严谨的药品说明书问答助手。必须用中文回答。"
                                "只能依据检索到的资料回答，不得补充资料外的医学知识。"
                                "涉及剂量、用法、禁忌、不良反应时必须保守，并提醒遵循医生或药师指导。"
                            ),
                        },
                        {"role": "user", "content": self._build_prompt(question, context)},
                    ],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": Config.LLM_TEMPERATURE,
                        "num_predict": Config.LLM_MAX_TOKENS,
                    },
                },
                timeout=Config.OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            answer = response.json().get("message", {}).get("content", "").strip()
            if not answer:
                answer = "本地模型没有返回有效答案。"
            verification = self.verifier.verify(answer, sources)
            return {
                "answer": answer,
                "sources": sources,
                "quality_score": self._assess_quality(answer, context),
                "confidence": self._calculate_confidence(context),
                "verification": verification,
                "response_time": time.time() - start_time,
            }
        except requests.exceptions.ConnectionError:
            message = f"无法连接 Ollama：{self.base_url}。请确认 Ollama 已启动。"
            logger.error(message)
            return self._error_response(message, start_time)
        except Exception as exc:
            logger.error(f"Ollama answer generation failed: {exc}")
            return self._error_response(f"生成答案失败：{exc}", start_time)

    def _build_prompt(self, question: str, context: List[Dict]) -> str:
        return f"""请根据下方检索资料回答用户问题。
硬性要求：
1. 全程使用中文回答。
2. 只能使用资料中的事实，不要补充资料外的医学知识。
3. 如果资料不足，明确回答“根据当前资料，未找到相关信息”。
4. 涉及剂量、用法、禁忌、不良反应时，必须保守表述。
5. 结尾列出引用，格式为：来源：资料名 / 页码或章节。
6. 最后加一句：具体用药请遵循医生或药师指导。

检索资料：
{self._format_context(context)}

用户问题：{question}
"""

    def _format_context(self, context: List[Dict]) -> str:
        parts = []
        for index, item in enumerate(context, 1):
            metadata = item.get("metadata", {}) or {}
            source = metadata.get("source") or metadata.get("source_file") or metadata.get("title") or "source"
            page = metadata.get("page") or metadata.get("section") or ""
            score = item.get("hybrid_score", item.get("score", 0))
            text = item.get("text", "")[:1500]
            parts.append(f"[资料{index}] 来源：{source} {page}；相关度：{score:.3f}\n{text}")
        return "\n\n".join(parts) if parts else "（当前没有检索资料）"

    def _extract_sources(self, context: List[Dict]) -> List[Dict]:
        sources = []
        for index, item in enumerate(context, 1):
            metadata = item.get("metadata", {}) or {}
            sources.append(
                {
                    "index": index,
                    "text": item.get("text", "")[:360],
                    "source": metadata.get("source") or metadata.get("source_file") or metadata.get("title") or "source",
                    "page": metadata.get("page", ""),
                    "section": metadata.get("section", ""),
                    "score": item.get("hybrid_score", item.get("score", 0)),
                }
            )
        return sources

    def _assess_quality(self, answer: str, context: List[Dict]) -> float:
        if not answer or not context:
            return 0.0
        return 0.8 if ("来源" in answer or "Source:" in answer) else 0.55

    def _calculate_confidence(self, context: List[Dict]) -> float:
        if not context:
            return 0.0
        scores = [float(item.get("hybrid_score", item.get("score", 0)) or 0) for item in context]
        return min(sum(scores) / len(scores), 1.0)

    def _error_response(self, message: str, start_time: float) -> Dict:
        return {
            "answer": message,
            "sources": [],
            "quality_score": 0,
            "confidence": 0,
            "response_time": time.time() - start_time,
            "verification": {
                "verified": False,
                "support_level": "none",
                "support_ratio": 0.0,
                "unsupported_sentences": [],
                "verification_note": message,
            },
            "error": message,
        }
