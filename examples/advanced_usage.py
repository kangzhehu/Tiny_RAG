#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/28 20:24
# @Author  : hukangzhe
# @File    : advanced_usage.py
# @Description :
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_pipeline import RAGPipeline, RAGEvaluator
from core.retriever import HybridRetriever, EnsembleRetriever, VectorRetriever
from core.document_loader import DirectoryLoader, Document
from core.utils import PerformanceMonitor, save_json, load_json  # 使用工具函数
import time


def advanced_rag_demo():
    """高级RAG使用示例"""

    # 初始化性能监控器
    monitor = PerformanceMonitor()

    # 1. 初始化RAG系统
    rag = RAGPipeline("../config.yaml")

    # 2. 批量加载文档
    print("批量加载文档...")

    # 模拟文档目录
    documents = [
        Document(
            content=f"这是关于{topic}的文档。{topic}是人工智能领域的重要技术。" * 10,
            metadata={"topic": topic, "doc_type": "technical"},
            doc_id=f"doc_{i}"
        )
        for i, topic in enumerate(["机器学习", "深度学习", "自然语言处理", "计算机视觉"])
    ]

    start_time = time.time()
    doc_ids = rag.add_documents(documents, show_progress=True)
    load_time = time.time() - start_time

    print(f"成功加载 {len(doc_ids)} 个文档片段，耗时: {load_time:.2f}秒")

    # 3. 测试不同的检索器
    print("\n测试不同的检索器配置...")

    retrievers_config = [
        ("向量检索器", VectorRetriever(
            rag.embedding_model,
            rag.vector_store,
            similarity_threshold=0.2
        )),
        ("混合检索器", HybridRetriever(
            rag.embedding_model,
            rag.vector_store,
            keyword_weight=0.3,
            similarity_threshold=0.2
        )),
        ("集成检索器", EnsembleRetriever([
            VectorRetriever(rag.embedding_model, rag.vector_store),
            HybridRetriever(rag.embedding_model, rag.vector_store)
        ], weights=[0.6, 0.4]))
    ]

    test_query = "机器学习和深度学习的关系"
    results_comparison = {}

    for name, retriever in retrievers_config:
        print(f"\n使用 {name}:")
        rag.set_retriever(retriever)

        # 测量性能
        start_time = time.time()
        retrieval_start = time.time()
        result = rag.query(test_query)
        retrieval_time = time.time() - retrieval_start
        total_time = time.time() - start_time

        # 记录性能
        monitor.record_query(total_time, retrieval_time, 0)

        print(f"  检索耗时: {retrieval_time:.3f}秒")
        print(f"  找到 {len(result['retrieved_chunks'])} 个相关文档")

        # 保存结果用于比较
        results_comparison[name] = {
            "num_results": len(result['retrieved_chunks']),
            "retrieval_time": retrieval_time,
            "top_similarity": result['retrieved_chunks'][0]['similarity'] if result['retrieved_chunks'] else 0
        }

    # 4. 批量查询测试
    print("\n\n执行批量查询测试...")
    questions = [
        "什么是机器学习？",
        "深度学习的应用有哪些？",
        "自然语言处理的挑战是什么？",
        "计算机视觉如何工作？"
    ]

    # 使用混合检索器
    rag.set_retriever(retrievers_config[1][1])

    batch_results = []
    start_time = time.time()

    for question in questions:
        query_start = time.time()
        result = rag.query(question)
        query_time = time.time() - query_start

        batch_results.append({
            "question": question,
            "answer_preview": result['answer'][:100] + "...",
            "num_sources": len(result.get('retrieved_chunks', [])),
            "query_time": query_time
        })

    batch_time = time.time() - start_time
    print(f"批量查询完成，总耗时: {batch_time:.2f}秒")
    print(f"平均每个查询: {batch_time / len(questions):.3f}秒")

    # 5. 评估系统
    print("\n\n评估RAG系统...")
    evaluator = RAGEvaluator(rag)

    # 准备测试数据
    test_queries = [
        {
            "question": "机器学习是什么？",
            "expected_docs": ["doc_0"],
            "filter_metadata": {"topic": "机器学习"}
        },
        {
            "question": "深度学习技术",
            "expected_docs": ["doc_1"],
            "filter_metadata": {"topic": "深度学习"}
        }
    ]

    # 评估检索质量
    retrieval_metrics = evaluator.evaluate_retrieval(test_queries)
    print("检索质量评估:")
    for metric, value in retrieval_metrics.items():
        print(f"  {metric}: {value:.3f}")

    # 6. 保存结果（使用utils中的函数）
    print("\n\n保存测试结果...")

    test_results = {
        "retrievers_comparison": results_comparison,
        "batch_query_results": batch_results,
        "retrieval_metrics": retrieval_metrics,
        "performance_stats": monitor.get_stats()
    }

    # 使用utils中的保存函数
    save_json(test_results, "test_results.json")
    print("结果已保存到 test_results.json")

    # 7. 测试缓存效果
    print("\n\n测试缓存效果...")

    # 第一次查询（无缓存）
    start_time = time.time()
    result1 = rag.query("机器学习的定义")
    time1 = time.time() - start_time

    # 第二次查询（有缓存）
    start_time = time.time()
    result2 = rag.query("机器学习的定义")
    time2 = time.time() - start_time

    print(f"首次查询耗时: {time1:.3f}秒")
    print(f"缓存查询耗时: {time2:.3f}秒")
    print(f"加速比: {time1 / time2:.2f}x")

    # 显示缓存统计
    cache_stats = rag.embedding_model.get_cache_stats()
    print(f"\n缓存统计:")
    print(f"  缓存文件数: {cache_stats['cache_files']}")
    print(f"  缓存大小: {cache_stats['cache_size_mb']:.2f} MB")


if __name__ == "__main__":
    print("=" * 60)
    print("高级RAG使用示例")
    print("=" * 60)

    advanced_rag_demo()