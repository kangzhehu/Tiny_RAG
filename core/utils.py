#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/29 10:44
# @Author  : hukangzhe
# @File    : utils.py
# @Description : 工具函数
import hashlib
import json
from typing import Any, Dict, List
import tiktoken
import re
from transformers import AutoTokenizer

# 全局 tokenizer 缓存，避免重复加载
_tokenizer_cache = {}


def count_tokens(text: str, model_name: str = "google/gemma-2-2b-it") -> int:
    """计算文本的token数量

    Args:
        text: 输入文本
        model_name: Hugging Face 模型名称（如 "google/gemma-2-2b-it"）

    Returns:
        token数量
    """
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)

    tokenizer = _tokenizer_cache[model_name]
    tokens = tokenizer.encode(text)
    return len(tokens)


def truncate_text(text: str, max_tokens: int,
                  model_name: str = "google/gemma-2-2b-it") -> str:
    """截断文本到指定token数量

    Args:
        text: 输入文本
        max_tokens: 最大token数
        model_name: Hugging Face 模型名称（如 "google/gemma-2-2b-it"）

    Returns:
        截断后的文本
    """
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)

    tokenizer = _tokenizer_cache[model_name]
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)

def count_tokens_gpt(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """计算文本的token数量

    Args:
        text: 输入文本
        model_name: 模型名称

    Returns:
        token数量
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def truncate_text_gpt(text: str, max_tokens: int,
                  model_name: str = "gpt-3.5-turbo") -> str:
    """截断文本到指定token数量

    Args:
        text: 输入文本
        max_tokens: 最大token数
        model_name: 模型名称

    Returns:
        截断后的文本
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def generate_hash(text: str) -> str:
    """生成文本的哈希值

    Args:
        text: 输入文本

    Returns:
        哈希值
    """
    return hashlib.md5(text.encode()).hexdigest()


def clean_text(text: str) -> str:
    """清理文本
    Args:
        text: 输入文本
    Returns:
        清理后的文本
    """
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)

    # 移除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:，。！？；：]', '', text)

    return text.strip()


def merge_metadata(metadata_list: List[Dict]) -> Dict:
    """合并多个元数据字典

    Args:
        metadata_list: 元数据列表

    Returns:
        合并后的元数据
    """
    merged = {}
    for metadata in metadata_list:
        for key, value in metadata.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list):
                merged[key].extend(value)
            elif isinstance(value, dict):
                merged[key].update(value)

    return merged


class JsonEncoder(json.JSONEncoder):
    """自定义JSON编码器"""

    def default(self, obj: Any) -> Any:
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def save_json(data: Any, file_path: str):
    """保存数据为JSON文件

    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=JsonEncoder)


def load_json(file_path: str) -> Any:
    """从JSON文件加载数据

    Args:
        file_path: 文件路径

    Returns:
        加载的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics = {
            "query_count": 0,
            "total_query_time": 0,
            "total_retrieval_time": 0,
            "total_generation_time": 0
        }

    def record_query(self, query_time: float, retrieval_time: float,
                     generation_time: float):
        """记录查询性能

        Args:
            query_time: 总查询时间
            retrieval_time: 检索时间
            generation_time: 生成时间
        """
        self.metrics["query_count"] += 1
        self.metrics["total_query_time"] += query_time
        self.metrics["total_retrieval_time"] += retrieval_time
        self.metrics["total_generation_time"] += generation_time

    def get_stats(self) -> Dict:
        """获取统计信息

        Returns:
            统计信息字典
        """
        if self.metrics["query_count"] == 0:
            return {"message": "No queries recorded"}

        count = self.metrics["query_count"]
        return {
            "query_count": count,
            "avg_query_time": self.metrics["total_query_time"] / count,
            "avg_retrieval_time": self.metrics["total_retrieval_time"] / count,
            "avg_generation_time": self.metrics["total_generation_time"] / count
        }