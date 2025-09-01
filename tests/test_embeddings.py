#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : hukangzhe
# @File    : test_embeddings.py
# @Description :
# tests/test_embeddings.py
import unittest
import numpy as np
import tempfile
import shutil
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embeddings import SentenceTransformerEmbedding


class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        """设置测试环境"""
        # 创建临时缓存目录
        self.temp_dir = tempfile.mkdtemp()
        self.embedding_model = SentenceTransformerEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=self.temp_dir,
            use_cache=True
        )

    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_single_text_embedding(self):
        """测试单个文本的向量化"""
        text = "This is a test sentence."
        embedding = self.embedding_model.embed(text)

        # 检查返回类型和形状
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (384,))  # all-MiniLM-L6-v2 的维度是384

    def test_multiple_texts_embedding(self):
        """测试多个文本的向量化"""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        embeddings = self.embedding_model.embed(texts)

        # 检查返回类型和形状
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (3, 384))

    def test_cache_functionality(self):
        """测试缓存功能"""
        text = "This text should be cached."

        # 第一次计算
        embedding1 = self.embedding_model.embed(text)

        # 第二次应该从缓存读取
        embedding2 = self.embedding_model.embed(text)

        # 检查两次结果是否相同
        np.testing.assert_array_almost_equal(embedding1, embedding2)

        # 检查缓存统计
        stats = self.embedding_model.get_cache_stats()
        self.assertGreater(stats['cache_files'], 0)

    def test_empty_input(self):
        """测试空输入"""
        embeddings = self.embedding_model.embed([])
        self.assertEqual(embeddings.shape, (0,))

    def test_dimension_property(self):
        """测试维度属性"""
        self.assertEqual(self.embedding_model.dimension, 384)


if __name__ == "__main__":
    unittest.main()