from typing import List, Dict, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from abc import ABC, abstractmethod
import jieba
import logging

logger = logging.getLogger(__name__)


class Retriever(ABC):
    """检索器基类"""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5,
                 filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """检索相关文档

        Args:
            query: 查询文本
            k: 返回结果数量
            filter_metadata: 元数据过滤条件

        Returns:
            检索结果列表
        """
        pass


class VectorRetriever(Retriever):
    """向量检索器"""

    def __init__(self, embedding_model, vector_store,
                 similarity_threshold: float = 0.0):
        """
        初始化向量检索器
        Args:
            embedding_model: 嵌入模型
            vector_store: 向量存储
            similarity_threshold: 相似度阈值
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str, k: int = 5,
                 filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """检索相关文档"""
        # 向量化查询
        query_vector = self.embedding_model.embed(query)
        if len(query_vector.shape) > 1:
            query_vector = query_vector[0]

        # 搜索相似文档
        texts, distances, metadatas = self.vector_store.search(
            query_vector, k, filter_metadata
        )

        # 计算相似度分数（将L2距离转换为相似度）
        # 使用 1/(1+distance) 将距离转换为相似度分数
        similarities_l2 = [1 / (1 + d) for d in distances]

        # 构建结果并过滤低相似度
        results = []
        for text, similarity, metadata in zip(texts, similarities_l2, metadatas):
            if similarity >= self.similarity_threshold:
                results.append({
                    "text": text,
                    "similarity": float(similarity),
                    "metadata": metadata,
                    "retriever_type": "vector"
                })

        logger.debug(f"Vector retriever found {len(results)} results above threshold {self.similarity_threshold}")
        return results


class HybridRetriever(Retriever):
    """混合检索器（向量+关键词）"""

    def __init__(self, embedding_model, vector_store,
                 keyword_weight: float = 0.3,
                 similarity_threshold: float = 0.0):
        """
        初始化混合检索器

        Args:
            embedding_model: 嵌入模型
            vector_store: 向量存储
            keyword_weight: 关键词权重 (0-1)
            similarity_threshold: 相似度阈值
        """
        self.vector_retriever = VectorRetriever(
            embedding_model,
            vector_store,
            similarity_threshold=similarity_threshold  # 混合检索器自己控制阈值
        )
        self.keyword_weight = keyword_weight
        self.similarity_threshold = similarity_threshold

        # 初始化jieba
        jieba.initialize()

    def retrieve(self, query: str, k: int = 5,
                 filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """混合检索"""

        # 1. 向量检索（获取更多结果用于重排序）
        vector_results = self.vector_retriever.retrieve(
            query,
            k=min(k * 3, 50),  # 获取3倍结果但不超过50个
            filter_metadata=filter_metadata
        )

        if not vector_results:
            return []

        # 2. 提取查询关键词
        query_terms = self._extract_keywords(query)

        # 3. 计算混合分数
        for result in vector_results:
            # 提取文档关键词
            doc_terms = self._extract_keywords(result["text"])

            # 计算关键词匹配分数
            if query_terms:
                # Jaccard相似度
                intersection = query_terms & doc_terms
                union = query_terms | doc_terms
                keyword_score = len(intersection) / len(union) if union else 0
            else:
                keyword_score = 0

            # 计算混合分数
            vector_score = result["similarity"]
            result["keyword_score"] = keyword_score
            result["hybrid_score"] = (
                    (1 - self.keyword_weight) * vector_score +
                    self.keyword_weight * keyword_score
            )
            result["retriever_type"] = "hybrid"

        # 4. 按混合分数排序并过滤
        vector_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # 5. 过滤低分结果
        filtered_results = [
            r for r in vector_results
            if r["hybrid_score"] >= self.similarity_threshold
        ]

        logger.debug(
            f"Hybrid retriever found {len(filtered_results)} results above threshold {self.similarity_threshold}")
        return filtered_results[:k]

    def _extract_keywords(self, text: str) -> set:
        """提取关键词"""
        # 使用jieba分词
        words = jieba.cut(text.lower())
        # 过滤停用词和短词
        keywords = {w for w in words if len(w) > 1}
        return keywords


class ReRankRetriever(Retriever):
    """重排序检索器"""

    def __init__(self, base_retriever: Retriever,
                 rerank_model=None,
                 similarity_threshold: float = 0.0):
        """
        初始化重排序检索器

        Args:
            base_retriever: 基础检索器
            rerank_model: 重排序模型（如交叉编码器）
            similarity_threshold: 相似度阈值
        """
        self.base_retriever = base_retriever
        self.rerank_model = rerank_model
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str, k: int = 5,
                 filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """检索并重排序"""

        # 1. 初步检索更多结果
        initial_k = min(k * 5, 100)  # 获取5倍结果但不超过100个
        initial_results = self.base_retriever.retrieve(
            query,
            k=initial_k,
            filter_metadata=filter_metadata
        )

        if not initial_results:
            return []

        # 2. 使用重排序模型
        if self.rerank_model:
            try:
                # 准备输入对
                pairs = [(query, r["text"]) for r in initial_results]

                # 获取重排序分数
                scores = self.rerank_model.predict(pairs)

                # 更新分数
                for result, score in zip(initial_results, scores):
                    result["rerank_score"] = float(score)
                    result["retriever_type"] = "rerank"

                # 按重排序分数排序
                initial_results.sort(key=lambda x: x["rerank_score"], reverse=True)

                # 过滤低分结果
                filtered_results = [
                    r for r in initial_results
                    if r.get("rerank_score", 0) >= self.similarity_threshold
                ]

            except Exception as e:
                logger.warning(f"Reranking failed: {e}. Using original scores.")
                filtered_results = initial_results
        else:
            # 没有重排序模型，直接使用原始结果
            filtered_results = initial_results

        logger.debug(f"Rerank retriever found {len(filtered_results)} results")
        return filtered_results[:k]


class EnsembleRetriever(Retriever):
    """集成检索器（组合多个检索器）"""

    def __init__(self, retrievers: List[Retriever],
                 weights: Optional[List[float]] = None):
        """
        初始化集成检索器

        Args:
            retrievers: 检索器列表
            weights: 各检索器权重
        """
        self.retrievers = retrievers
        if weights:
            assert len(weights) == len(retrievers)
            self.weights = weights
        else:
            # 默认等权重
            self.weights = [1.0 / len(retrievers)] * len(retrievers)

    def retrieve(self, query: str, k: int = 5,
                 filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """集成检索"""

        all_results = {}

        # 从每个检索器获取结果
        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.retrieve(query, k=k * 2, filter_metadata=filter_metadata)

            for result in results:
                # 使用文本作为key来合并结果
                key = result["text"][:100]  # 使用前100字符作为key

                if key not in all_results:
                    all_results[key] = result
                    all_results[key]["ensemble_score"] = 0

                # 累加加权分数
                score = result.get("hybrid_score", result.get("similarity", 0))
                all_results[key]["ensemble_score"] += weight * score

        # 转换为列表并排序
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x["ensemble_score"], reverse=True)

        # 标记检索器类型
        for result in final_results:
            result["retriever_type"] = "ensemble"

        return final_results[:k]