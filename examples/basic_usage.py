#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/28 20:22
# @Author  : hukangzhe
# @File    : basic_usage.py
# @Description : 简单测试，一个基础的使用示例

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_pipeline import RAGPipeline
from core.document_loader import Document
from core.utils import count_tokens, clean_text  # 使用工具函数


def main():
    # 1. 初始化RAG系统
    print("初始化RAG系统...")
    rag = RAGPipeline(config_path="../config.yaml")

    # 2. 添加文档
    print("添加文档到知识库...")

    # 使用工具函数清理文本
    raw_text = """
        RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的技术。
        它首先从知识库中检索相关信息，然后基于这些信息生成回答。
        这种方法可以有效解决大语言模型的知识时效性问题和幻觉问题。
    """
    # 清理文本， 移除多余空白字符和特殊字符
    cleaned_text = clean_text(raw_text)  # 使用utils中的清理函数
    token_count = count_tokens(cleaned_text)  # 使用utils中的token计数

    print(f"文档token数: {token_count}")

    doc = Document(
        content=cleaned_text,
        metadata={"source": "rag_intro.txt", "topic": "RAG", "tokens": token_count},
        doc_id="doc_001"
    )
    rag.add_documents([doc])

    # 3. 查询系统
    print("\n开始查询...")
    questions = [
        "什么是RAG技术？",
        "RAG可以解决什么问题？",
        "RAG的工作原理是什么？"
    ]

    for question in questions:
        print(f"\n问题: {question}")

        # 基本查询
        result = rag.query(question)
        print(f"答案: {result['answer'][:200]}...")

        if result.get('retrieved_chunks'):
            print(f"相关度最高的文档片段:")
            for i, chunk in enumerate(result['retrieved_chunks'][:2], 1):
                print(f"  {i}. {chunk['text'][:100]}...")
                print(f"     相似度: {chunk['similarity']:.3f}")
                print(f"     来源: {chunk['metadata'].get('source', 'unknown')}")

    # 4. 带过滤的查询
    print("\n\n带过滤条件的查询...")
    result = rag.query(
        "RAG技术",
        filter_metadata={"topic": "RAG"}
    )
    print(f"找到 {len(result['retrieved_chunks'])} 个相关文档")

    # 5. 查看统计信息
    print("\n系统统计信息:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

