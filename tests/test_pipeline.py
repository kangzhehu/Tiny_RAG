#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : hukangzhe
# @File    : test_pipeline.py
# @Description : 基础测试
import unittest

from pure_eval import Evaluator

from core.document_loader import Document
from core.rag_pipeline import RAGPipeline, RAGEvaluator


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        """ 设置测试环境 """
        self.rag_pipeline = RAGPipeline()

        # 添加测试文档
        test_docs = [
            Document(
                content="人工智能是计算机科学的一个分支，致力于创建智能机器。",
                metadata={"source": "test1.txt", "category": "AI"},
                doc_id="test_doc_1"
            ),
            Document(
                content="深度学习是机器学习的一个子领域，使用神经网络处理数据。",
                metadata={"source": "test2.txt", "category": "ML"},
                doc_id="test_doc_2"
            ),
            Document(
                content="自然语言处理让计算机理解人类语言。",
                metadata={"source": "test3.txt", "category": "NLP"},
                doc_id="test_doc_3"
            )
        ]
        
        self.rag_pipeline.add_documents(test_docs)

    def test_query_basic(self):
        """ 测试基础查询 """
        result = self.rag_pipeline.query("什么是人工智能")
        self.assertIsNotNone(result)
        self.assertIn("answer", result)
        self.assertIn("retrieved_chunks", result)

    def test_query_with_filter(self):
        """测试带过滤的查询"""
        result = self.rag_pipeline.query(
            "机器学习相关内容",
            filter_metadata={"category": "ML"}
        )

        self.assertIsNotNone(result)
        # 验证只返回ML类别的文档
        if result["retrieved_chunks"]:
            for chunk in result["retrieved_chunks"]:
                self.assertEqual(chunk["metadata"].get("category"), "ML")



    def test_evaluator(self):
        """ 测试评估 """
        evaluator = RAGEvaluator(self.rag_pipeline)
        test_queries = [
            {
                "question": "什么是人工智能？",
                "expected_docs": ["test_doc_1"]
            },
            {
                "question": "深度学习是什么？",
                "expected_docs": ["test_doc_2"],
                "filter_metadata": {"category": "ML"}
            }
        ]
        metrics = evaluator.evaluate_retrieval(test_queries)
        self.assertIn("avg_precision", metrics)
        self.assertIn("avg_recall", metrics)
        self.assertIn("avg_f1", metrics)
        self.assertIn("avg_mrr", metrics)
        self.assertGreaterEqual(metrics["avg_precision"], 0)
        self.assertLessEqual(metrics["avg_precision"], 1)


    def test_stats(self):
        """ 测试统计功能 """
        stats = self.rag_pipeline.get_stats()

        self.assertIn("total_documents", stats)
        self.assertGreater(stats["total_documents"], 0)



if __name__ == '__main__':
    unittest.main()