#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility helpers for local PyTorch installs."""


def patch_torch_metadata_version() -> None:
    """Let transformers recover when torch package metadata has an empty version."""
    try:
        import importlib.metadata as metadata
    except Exception:
        return

    original_version = metadata.version

    def version(package_name: str):
        value = original_version(package_name)
        if package_name.lower() == "torch" and not value:
            try:
                import torch

                return torch.__version__
            except Exception:
                return value
        return value

    metadata.version = version
