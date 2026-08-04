#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Question-answering pipeline for the full RAG stack."""

import time
from typing import Dict, Optional

from generation.answer_generator import AnswerGenerator
from retrieval.hybrid_retriever import HybridRetriever
from utils.logger import logger


class QAPipeline:
    """Combine BGE-M3/Milvus/BM25 retrieval, rerank, and Ollama generation."""

    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()
        logger.info("Full RAG QA pipeline initialized")

    def answer(self, question: str, user_context: Optional[Dict] = None) -> Dict:
        start_time = time.time()
        try:
            retrieved_context = self.retriever.retrieve(question)
            if not retrieved_context:
                return {
                    "answer": "根据当前资料，未找到相关信息。请先导入药品目录或说明书数据。",
                    "sources": [],
                    "source_type": "none",
                    "confidence": 0,
                    "verification": {
                        "verified": False,
                        "unsupported_sentences": [],
                        "verification_note": "没有可核验的来源片段。",
                    },
                    "response_time": time.time() - start_time,
                }

            result = self.generator.generate(question, retrieved_context)
            result["response_time"] = time.time() - start_time
            result["source_type"] = "rag"
            return result
        except Exception as exc:
            logger.error(f"QA processing failed: {exc}")
            return {
                "answer": f"处理问题时出错：{exc}",
                "sources": [],
                "source_type": "error",
                "confidence": 0,
                "verification": {
                    "verified": False,
                    "unsupported_sentences": [],
                    "verification_note": str(exc),
                },
                "response_time": time.time() - start_time,
                "error": str(exc),
            }
