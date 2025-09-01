#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/21 15:23
# @Author  : hukangzhe
# @File    : embeddings.py
# @Description : 向量化模块
import hashlib
from typing import Union, List
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """文本向量化"""
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """向量维度"""
        raise NotImplementedError


class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir: str = "./cache/embeddings",
                 use_cache: bool = True,
                 batch_size: int = 32,):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.embedder = SentenceTransformer(model_name)

        # 创建缓存目录
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)

        # 获取向量维度
        self._dimension = self.embedder.get_sentence_embedding_dimension()

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        # 确保输入是列表
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        # 如果不使用缓存，直接计算
        if not self.use_cache:
            embeddings = self.embedder.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts)>100,
                batch_size=self.batch_size,
            )
            return embeddings[0] if is_single else embeddings

        # 使用缓存
        embeddings = []
        uncache_texts = []
        uncached_indices = []

        # 检查缓存
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cache_embedding = self._load_from_cache(cache_key)

            if cache_embedding is not None:
                embeddings.append(cache_embedding)
            else:
                embeddings.append(None) # 占位
                uncached_indices.append(i)
                uncache_texts.append(text)

        # 批量计算未缓存的向量
        if uncache_texts:
            new_embeddings = self.embedder.encode(
                uncache_texts,
                convert_to_numpy=True,
                convert_to_tensor=False,
                batch_size=self.batch_size
            )

            # 保存到缓存中
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                cache_key = self._get_cache_key(texts[idx])
                self._save_to_cache(cache_key, embedding)

        result = np.array(embeddings)
        # 如果输入单个字符串，返回单个向量
        return result[0] if is_single else result

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_cache_key(self, text: str) -> str:
        """
        生成缓存键
        Args:
            text: 输入文本

        Returns:
            缓存键
        """
        # 结合模型名和文本内容生成唯一键
        content = f"{self.model_name}:{text}"
        text_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return text_hash

    def _get_cache_path(self, cache_key: str) -> str:
        """
        获取缓存文件路径
        Args:
            cache_key: 缓存键
        Returns:
            缓存文件路径
        """
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _load_from_cache(self, cache_key: str) -> np.ndarray:
        """
        从缓存加载向量

        Args:
            cache_key: 缓存键

        Returns:
            缓存的向量，如果不存在返回None
        """
        cache_path = self._get_cache_path(cache_key)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
                # 删除损坏的缓存文件
                os.remove(cache_path)

        return None

    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """
        保存向量到缓存

        Args:
            cache_key: 缓存键
            embedding: 向量
        """
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")

    def clear_cache(self):
        """清空缓存"""
        if os.path.exists(self.cache_dir):
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("Cache cleared")

    def get_cache_stats(self) -> dict:
        """
        获取缓存统计信息
        Returns:
            缓存统计信息
        """
        if not os.path.exists(self.cache_dir):
            return {"cache_files": 0, "cache_size_mb": 0}

        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f))
            for f in cache_files
        )

        return {
            "cache_files": len(cache_files),
            "cache_size_mb": total_size / (1024 * 1024)
        }


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI向量化模型"""

    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002",
                 max_retries: int = 3):
        """
        初始化OpenAI嵌入模型

        Args:
            api_key: API密钥
            model_name: 模型名称
            max_retries: 最大重试次数
        """
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries

        # 设置向量维度
        self._dimension = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }.get(model_name, 1536)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        调用OpenAI API进行向量化

        Args:
            texts: 单个文本或文本列表

        Returns:
            向量数组
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        import time

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )

                embeddings = [item.embedding for item in response.data]
                result = np.array(embeddings)

                return result[0] if is_single else result

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Embedding failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise e

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return self._dimension