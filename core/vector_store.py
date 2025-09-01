#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/24 16:47
# @Author  : hukangzhe
# @File    : vector_store.py
# @Description :
import os
import pickle

import faiss
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple


@dataclass
class VectorData:
    """向量数据类"""
    vector: np.ndarray
    text: str
    metadata: Dict[str, Any]
    doc_id: str


class VectorStore:

    def add(self, vectors: np.ndarray, texts: List[str],
            metadatas: List[Dict] = None, doc_ids: List[str] = None) -> List[str]:
        """添加向量"""
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int = 5, filter_metadata: Dict = None) -> Tuple[List[str], List[float], List[Dict]]:
        """搜索相似向量"""
        raise NotImplementedError

    def save(self, path: str):
        """保存索引"""
        raise NotImplementedError

    def load(self, path: str):
        """加载索引"""
        raise NotImplementedError


class FaissVectorStore(VectorStore):
    def __init__(self, dimension: int, index_type: str = "Flat"):
        self.dimension = dimension
        self.index_type = index_type

        # 初始化Faiss index
        if index_type == "Flat":
            # 计算查询向量和所有向量L2
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVF":
            # 先用 k-means 把数据库向量聚类成多个簇（每个簇一个 centroid）。
            # 搜索时只在与查询向量最近的几个簇中做 FlatL2 搜索，减少计算量。
            quantizer = faiss.IndexFlatL2(dimension)
            # 100是聚类中心数
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # 存储文本和元数据
        self.texts = []
        self.metadatas = []
        self.doc_ids = []

    def add(self, vectors: np.ndarray, texts: List[str],
            metadatas: List[Dict] = None, doc_ids: List[str] = None) -> List[str]:
        """添加向量到索引"""

        # 确保向量是float32类型
        vectors = np.array(vectors).astype('float32')

        # 如果是IVF索引且未训练，先训练
        if self.index_type == "IVF" and not self.index.is_trained:
            self.index.train(vectors)

        # 向量添加到索引
        self.index.add(vectors)

        # 存储文本和元数据
        self.texts.extend(texts)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(texts))  # [{}, ...{}]

        # 向向量库里添加新文档的时候，给每个文档生成唯一的 ID
        if doc_ids:
            self.doc_ids.extend(doc_ids)  # 调用add时，用户传入doc_ids，则使用
        else:
            start_id = len(self.doc_ids)  # 如果用户没有传入，则生成新的
            new_ids = [f"doc_{start_id + i}" for i in range(len(texts))]
            self.doc_ids.extend(new_ids)

        return self.doc_ids[-len(texts):]

    def search(self, query_vector: np.ndarray, k: int = 5, filter_metadata: Dict = None) -> Tuple[List[str], List[float], List[Dict]]:
        """搜索与query最相似的K个向量"""

        # 确保查询向量是正确的形状和类型
        query_vector = np.array(query_vector).reshape(1, -1).astype('float32')
        # 当前已添加的向量总数如果少于K，搜索最小的个数
        distance, indices = self.index.search(query_vector, min(k, self.index.ntotal))

        # 获取结果
        results_texts = []
        results_distances = []
        results_metadatas = []

        for i, idx in enumerate(indices[0]):
            if idx >= 0:    # -1表示没有找到
                # 应用元数据过滤
                if filter_metadata:
                    metadata = self.metadatas[idx]
                    if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                        continue

                results_texts.append(self.texts[idx])
                results_distances.append(float(distance[0][i]))
                results_metadatas.append(self.metadatas[idx])

        return results_texts, results_distances, results_metadatas

    def delete(self, doc_ids: List[str]):
        """ 删除指定id文档 """
        # FAISS不直接支持删除，需要重建索引
        indices_to_keep = [i for i, doc_id in enumerate(self.doc_ids)
                           if doc_id not in doc_ids]

        if len(indices_to_keep) < len(self.doc_ids):
            # 获取要保留的向量
            vectors_to_keep = []
            for i in indices_to_keep:
                vector = self.index.reconstruct(i)
                vectors_to_keep.append(vector)

            # 重建索引
            self.index.reset()
            if vectors_to_keep:
                vectors_to_keep = np.array(vectors_to_keep)
                if self.index_type == "IVF":
                    self.index.train(vectors_to_keep)
                self.index.add(vectors_to_keep)

            # 更新元数据
            self.texts = [self.texts[i] for i in indices_to_keep]
            self.metadatas = [self.metadatas[i] for i in indices_to_keep]
            self.doc_ids = [self.doc_ids[i] for i in indices_to_keep]

    def save(self, path: str):
        """ 保存索引和元数据 """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 保存FAISS索引
        faiss.write_index(self.index, path+'.index')

        # 保存元数据
        metadata = {
            "texts": self.texts,
            "metadatas": self.metadatas,
            "doc_ids": self.doc_ids,
            "dimension": self.dimension,
            "index_type": self.index_type
        }
        with open(path + ".meta", 'wb') as f:
            pickle.dump(metadata, f)

    def load(self, path: str):
        """加载索引和元数据"""
        # 加载FAISS索引
        self.index = faiss.read_index(path + ".index")

        # 加载元数据
        with open(path + ".meta", 'rb') as f:
            metadata = pickle.load(f)

        self.texts = metadata["texts"]
        self.metadatas = metadata["metadatas"]
        self.doc_ids = metadata["doc_ids"]
        self.dimension = metadata["dimension"]
        self.index_type = metadata["index_type"]

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }