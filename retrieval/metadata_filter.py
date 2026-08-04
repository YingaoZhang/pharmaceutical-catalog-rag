#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metadata-aware filtering for drug-specific questions."""

import json
import os
import re
from typing import Dict, List, Tuple

from config import Config
from utils.logger import logger


SECTION_QUERY_TERMS = (
    (
        "\u7528\u6cd5\u7528\u91cf",
        (
            "\u7528\u6cd5",
            "\u7528\u91cf",
            "\u5242\u91cf",
            "\u600e\u4e48\u5403",
            "\u600e\u4e48\u7528",
            "\u670d\u7528",
            "\u7ed9\u836f",
        ),
    ),
    ("\u7981\u5fcc", ("\u7981\u5fcc", "\u4e0d\u80fd\u7528", "\u7981\u7528", "\u5fcc\u7528")),
    (
        "\u4e0d\u826f\u53cd\u5e94",
        ("\u4e0d\u826f\u53cd\u5e94", "\u526f\u4f5c\u7528", "\u4e0d\u826f", "\u53cd\u5e94"),
    ),
    (
        "\u6ce8\u610f\u4e8b\u9879",
        ("\u6ce8\u610f\u4e8b\u9879", "\u6ce8\u610f", "\u614e\u7528", "\u8b66\u544a"),
    ),
    (
        "\u9002\u5e94\u75c7",
        ("\u9002\u5e94\u75c7", "\u9002\u7528\u4e8e", "\u6cbb\u7597\u4ec0\u4e48", "\u4e3b\u6cbb", "\u529f\u80fd\u4e3b\u6cbb"),
    ),
    (
        "\u7279\u6b8a\u4eba\u7fa4\u7528\u836f",
        ("\u5b55\u5987", "\u54fa\u4e73", "\u513f\u7ae5", "\u8001\u4eba", "\u8001\u5e74"),
    ),
    (
        "\u836f\u7269\u76f8\u4e92\u4f5c\u7528",
        ("\u76f8\u4e92\u4f5c\u7528", "\u5408\u7528", "\u8054\u7528", "\u5e76\u7528"),
    ),
    (
        "\u836f\u7406\u6bd2\u7406",
        ("\u836f\u7406", "\u6bd2\u7406"),
    ),
    (
        "\u836f\u4ee3\u52a8\u529b\u5b66",
        ("\u836f\u4ee3", "\u52a8\u529b\u5b66"),
    ),
    (
        "\u8d2e\u85cf",
        ("\u8d2e\u85cf", "\u50a8\u5b58", "\u4fdd\u5b58"),
    ),
    (
        "\u6709\u6548\u671f",
        ("\u6709\u6548\u671f", "\u4fdd\u8d28\u671f"),
    ),
)


class MetadataFilter:
    """Find drug-specific chunks before semantic search."""

    def __init__(self):
        self.documents: List[Dict] = []
        self.drug_terms: List[str] = []
        self.reload()

    def reload(self) -> None:
        self.documents = self._load_documents()
        terms = set()
        for doc in self.documents:
            metadata = doc.get("metadata", {}) or {}
            for key in (
                "drug_name",
                "generic_name",
                "trade_name",
                "title",
                "approval_number",
            ):
                value = str(metadata.get(key, "")).strip()
                if value:
                    terms.add(value.lower())
                    for token in re.split(r"[\s,;()/]+", value.lower()):
                        if len(token) >= 4:
                            terms.add(token)
        self.drug_terms = sorted(terms, key=len, reverse=True)
        logger.info(f"Metadata filter loaded {len(self.drug_terms)} drug/title terms")

    def filter(self, query: str, limit: int = None) -> Tuple[List[Dict], List[str]]:
        if not Config.ENABLE_METADATA_FILTER:
            return [], []
        limit = limit or Config.METADATA_FILTER_TOP_K
        query_lower = query.lower()
        matched_terms = [term for term in self.drug_terms if term and term in query_lower]
        if not matched_terms:
            return [], []

        matched_sections = self._matched_sections(query_lower)
        filtered = []
        for doc in self.documents:
            haystack = self._doc_haystack(doc)
            if any(term in haystack for term in matched_terms):
                if matched_sections and not self._section_matches(doc, matched_sections):
                    continue
                metadata = doc.get("metadata", {}) or {}
                enriched = doc.copy()
                enriched["drug_name"] = metadata.get("drug_name", enriched.get("drug_name", ""))
                section_bonus = self._section_bonus(doc, matched_sections)
                enriched["metadata_score"] = 1.0 + section_bonus
                enriched["source"] = "metadata"
                filtered.append(enriched)
        filtered.sort(key=lambda item: item.get("metadata_score", 0), reverse=True)
        return filtered[:limit], matched_terms

    def _doc_haystack(self, doc: Dict) -> str:
        metadata = doc.get("metadata", {}) or {}
        parts = [
            metadata.get("drug_name", ""),
            metadata.get("generic_name", ""),
            metadata.get("trade_name", ""),
            metadata.get("title", ""),
            metadata.get("approval_number", ""),
            metadata.get("manufacturer", ""),
            metadata.get("related_disease", ""),
            metadata.get("section_name", ""),
            metadata.get("field_name", ""),
            metadata.get("source", ""),
            doc.get("text", "")[:500],
        ]
        return " ".join(str(part).lower() for part in parts)

    def _matched_sections(self, query_lower: str) -> List[str]:
        matched = []
        for section_name, terms in SECTION_QUERY_TERMS:
            if any(term in query_lower for term in terms):
                matched.append(section_name.lower())
        return matched

    def _section_matches(self, doc: Dict, matched_sections: List[str]) -> bool:
        metadata = doc.get("metadata", {}) or {}
        section_text = " ".join(
            [
                str(metadata.get("section_name", "")),
                str(metadata.get("field_name", "")),
                str(metadata.get("section", "")),
            ]
        ).lower()
        return any(section in section_text for section in matched_sections)

    def _section_bonus(self, doc: Dict, matched_sections: List[str]) -> float:
        if not matched_sections:
            return 0.0
        metadata = doc.get("metadata", {}) or {}
        section_text = " ".join(
            [
                str(metadata.get("section_name", "")),
                str(metadata.get("field_name", "")),
                str(metadata.get("section", "")),
                str(doc.get("text", "")[:300]),
            ]
        ).lower()
        return 0.6 if any(section in section_text for section in matched_sections) else 0.0

    def _load_documents(self) -> List[Dict]:
        if not os.path.exists(Config.LOCAL_DOC_STORE):
            return []
        try:
            with open(Config.LOCAL_DOC_STORE, "r", encoding="utf-8") as file:
                first_char = file.read(1)
                file.seek(0)
                if first_char == "[" or first_char == "{":
                    try:
                        payload = json.load(file)
                        return payload if isinstance(payload, list) else payload.get("documents", [])
                    except json.JSONDecodeError:
                        file.seek(0)
                return [json.loads(line) for line in file if line.strip()]
        except Exception as exc:
            logger.warning(f"Failed to load local documents for metadata filtering: {exc}")
            return []
