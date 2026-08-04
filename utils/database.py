#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Database clients for Redis and Milvus."""

from typing import Optional

import redis
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient as PymilvusClient,
    connections,
    utility,
)

from config import Config
from utils.logger import logger


class RedisClient:
    """Small Redis wrapper used for retrieval cache."""

    def __init__(self):
        self.client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            password=Config.REDIS_PASSWORD or None,
            db=Config.REDIS_DB,
            decode_responses=True,
        )
        self.client.ping()
        logger.info("Redis connected")

    def get(self, key: str) -> Optional[str]:
        return self.client.get(key)

    def set(self, key: str, value: str, ex: int = None):
        return self.client.set(key, value, ex=ex)

    def setex(self, key: str, value: str, ex: int):
        return self.client.setex(key, ex, value)

    def delete(self, key: str):
        return self.client.delete(key)

    def exists(self, key: str):
        return self.client.exists(key)


class MilvusClient:
    """Milvus collection helper."""

    def __init__(self):
        try:
            if connections.has_connection("default"):
                logger.info("Milvus connection reused")
                return
        except Exception:
            pass
        connections.connect(
            alias="default",
            host=Config.MILVUS_HOST,
            port=Config.MILVUS_PORT,
        )
        logger.info("Milvus connected")

    def collection_exists(self, collection_name: str) -> bool:
        return utility.has_collection(collection_name)

    def create_collection(self, collection_name: str, dim: int) -> Collection:
        if utility.has_collection(collection_name):
            return Collection(collection_name)

        client = PymilvusClient(uri=f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}")
        schema = client.create_schema(enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="drug_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=8192,
            enable_analyzer=True,
            analyzer_params={"tokenizer": "icu"},
        )
        schema.add_field(field_name="language", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_function(
            Function(
                name="text_to_sparse_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 1024},
        )
        index_params.add_index(field_name="sparse", index_type="AUTOINDEX", metric_type="BM25")
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        collection = Collection(name=collection_name)
        logger.info(f"Milvus collection created: {collection_name}")
        return collection

    def get_collection(self, collection_name: str) -> Collection:
        return Collection(collection_name)
