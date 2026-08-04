#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch import Excel drug instructions into the RAG index."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import EXCEL_SUFFIXES, _documents_from_excel, _index_documents


def discover_excel_files(paths):
    files = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for suffix in EXCEL_SUFFIXES:
                files.extend(sorted(path.glob(f"*{suffix}")))
        elif path.is_file() and path.suffix.lower() in EXCEL_SUFFIXES:
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import .xlsx/.xlsm/.xls files into Milvus native sparse BM25."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Excel file paths or directories containing Excel files.",
    )
    args = parser.parse_args()

    files = discover_excel_files(args.paths)
    if not files:
        print("No Excel files found.")
        return 1

    total_chunks = 0
    for path in files:
        documents = _documents_from_excel(path.read_bytes(), path.name)
        added = _index_documents(documents)
        total_chunks += added
        print(f"{path}: {added} chunks indexed")

    print(f"Done. {len(files)} files imported, {total_chunks} chunks indexed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
