#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entrypoint for the retired local BM25 pickle index."""


def main() -> int:
    print(
        "Local rank_bm25 pickle indexing has been retired. "
        "BM25 sparse vectors are generated inside Milvus from the text field.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
