#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-generation verification to reduce unsupported medical claims."""

import re
from typing import Dict, List

import jieba

from config import Config


class AnswerVerifier:
    """Estimate whether an answer is grounded in retrieved source snippets."""

    def verify(self, answer: str, sources: List[Dict]) -> Dict:
        if not answer or not sources:
            return {
                "verified": False,
                "support_level": "none",
                "support_ratio": 0.0,
                "unsupported_sentences": [],
                "verification_note": "未找到可用于核验的来源片段。",
            }

        source_text = self._normalize(" ".join(source.get("text", "") for source in sources))
        sentences = [s for s in self._split_sentences(answer) if not self._is_boilerplate(s)]
        if not sentences:
            return {
                "verified": True,
                "support_level": "high",
                "support_ratio": 1.0,
                "unsupported_sentences": [],
                "verification_note": "回答主要由固定用药提醒组成，未发现明显脱离来源的医学断言。",
            }

        sentence_scores = [(sentence, self._support_score(sentence, source_text)) for sentence in sentences]
        unsupported = [sentence for sentence, score in sentence_scores if score < 0.18]
        support_ratio = sum(score for _, score in sentence_scores) / len(sentence_scores)

        verified = (
            len(sources) >= Config.MIN_SUPPORTED_SOURCES
            and support_ratio >= 0.30
            and len(unsupported) <= max(1, len(sentences) // 2)
        )
        support_level = self._support_level(verified, support_ratio, len(unsupported), len(sentences))
        note = self._verification_note(support_level)

        return {
            "verified": verified,
            "support_level": support_level,
            "support_ratio": round(support_ratio, 3),
            "unsupported_sentences": unsupported[:5],
            "verification_note": note,
        }

    def _split_sentences(self, text: str) -> List[str]:
        cleaned = re.sub(r"[*#>`]+", "", text)
        cleaned = re.sub(r"^\s*\d+[.、]\s*", "", cleaned, flags=re.MULTILINE)
        parts = re.split(r"(?<=[。！？!?])\s*|\n+", cleaned)
        return [part.strip(" -\t") for part in parts if len(part.strip(" -\t")) >= 12]

    def _support_score(self, sentence: str, source_text: str) -> float:
        normalized = self._normalize(sentence)
        if not normalized:
            return 1.0
        if normalized in source_text:
            return 1.0

        tokens = self._tokens(normalized)
        if not tokens:
            return 1.0
        hits = sum(1 for token in tokens if token in source_text)
        token_score = hits / len(tokens)

        bigrams = self._chinese_bigrams(normalized)
        if not bigrams:
            return token_score
        bigram_hits = sum(1 for token in bigrams if token in source_text)
        bigram_score = bigram_hits / len(bigrams)
        return max(token_score, bigram_score * 0.75)

    def _tokens(self, text: str) -> List[str]:
        stopwords = {
            "根据",
            "资料",
            "检索",
            "显示",
            "如下",
            "此外",
            "患者",
            "情况",
            "相关",
            "需要",
            "特别",
            "谨慎",
            "明确",
            "列为",
        }
        tokens = []
        tokens.extend(re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower()))
        for token in jieba.cut(text):
            token = token.strip().lower()
            if len(token) >= 2 and re.search(r"[\u4e00-\u9fffA-Za-z0-9]", token) and token not in stopwords:
                tokens.append(token)
        return tokens

    def _chinese_bigrams(self, text: str) -> List[str]:
        chars = re.findall(r"[\u4e00-\u9fff]", text)
        if len(chars) < 2:
            return []
        return ["".join(chars[index : index + 2]) for index in range(len(chars) - 1)]

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", text.lower())

    def _support_level(
        self,
        verified: bool,
        support_ratio: float,
        unsupported_count: int,
        sentence_count: int,
    ) -> str:
        if verified and support_ratio >= 0.46 and unsupported_count == 0:
            return "high"
        if verified:
            return "medium"
        if sentence_count and unsupported_count < sentence_count:
            return "review"
        return "low"

    def _verification_note(self, support_level: str) -> str:
        notes = {
            "high": "来源支持度高，回答内容与检索片段匹配良好。",
            "medium": "来源支持度较高，回答经过概括整理；建议同时查看来源片段。",
            "review": "部分表述是模型概括结果，建议结合下方来源片段复核关键用药信息。",
            "low": "当前回答与来源片段匹配不足，请优先以下方来源片段为准。",
            "none": "未找到可用于核验的来源片段。",
        }
        return notes.get(support_level, notes["review"])

    def _is_boilerplate(self, sentence: str) -> bool:
        lowered = sentence.lower().strip()
        boilerplate_prefixes = (
            "来源",
            "引用",
            "具体用药",
            "临床用药",
            "仅供参考",
            "source:",
        )
        if lowered.startswith(boilerplate_prefixes):
            return True
        return len(lowered) <= 30 and any(
            item in lowered
            for item in (
                "医生",
                "药师",
                "follow a doctor",
                "pharmacist",
            )
        )
