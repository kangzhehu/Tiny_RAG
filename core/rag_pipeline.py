#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/26 9:02
# @Author  : hukangzhe
# @File    : rag_pipeline.py
# @Description : RAG主流程
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

from sentence_transformers import CrossEncoder

from core.document_loader import Document, DocumentLoader, DirectoryLoader
from core.embeddings import SentenceTransformerEmbedding
from core.llm_handler_gemma import LocalLLMHandler
from core.retriever import HybridRetriever, ReRankRetriever, Retriever, VectorRetriever
from core.text_splitter import RecursiveCharacterTextSplitter
from core.vector_store import FaissVectorStore

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """ RAG完整流程管道 """
    def __init__(self, config_path: str = None):
        """
        初始化管道
        :param config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        print(self.config)
        self.embedding_model = None
        self.text_splitter = None
        self.vector_store = None
        self.retriever = None
        self.llm_handler = None
        self.reranker = None    # 检索 + 重排器
        self._initialize_components()

    def _load_config(self, config_path: str) -> Dict:
        """ 加载配置文件 """
        logging.info("Loading config... ")
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 默认配置
            return {
                "embedding": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimension": 384
                },
                "text_splitter": {
                    "chunk_size": 500,
                    "chunk_overlap": 50
                },
                "vector_store": {
                    "type": "faiss",
                    "index_type": "Flat",
                    "index_path": "./data/vector_db/index.pkl"
                },
                "retriever": {
                    "top_k": 3,
                    "similarity_threshold": 0.6
                },
                "llm": {
                    "model_name": "google/gemma-2-2b-it",
                    "temperature": 0.7,
                    "max_new_tokens": 1000
                },
                "reranker": {
                    "model_name": "BAAI/bge-reranker-base",
                }

            }

    def _initialize_components(self):
        """ 初始化所有RAG组件 """
        logging.info("Initializing RAG components...")
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformerEmbedding(model_name=self.config["embedding"]["model_name"],
                                                            cache_dir=self.config["embedding"]["cache_dir"],
                                                            use_cache=self.config["embedding"]["use_cache"],
                                                            batch_size=self.config["embedding"]["batch_size"],)
        # 初始化向量存储
        self.vector_store = FaissVectorStore(dimension=self.config["embedding"]["dimension"],
                                             index_type=self.config["vector_store"]["index_type"])
        # 尝试加载已有的索引
        index_path = self.config["vector_store"]["index_path"]
        if Path(f"{index_path}.index").exists():
            logging.info(f"Loading index from {index_path}.index ...")
            self.vector_store.load(index_path)

        # 根据配置选择检索器类型
        retriever_type = self.config["retriever"].get("type", "vector")

        if retriever_type == "hybrid":
            self.retriever = HybridRetriever(
                embedding_model=self.embedding_model,
                vector_store=self.vector_store,
                keyword_weight=self.config["retriever"].get("keyword_weight", 0.3),
                similarity_threshold=self.config["retriever"]["similarity_threshold"]
            )
        elif retriever_type == "rerank":
            # 先创建基础检索器
            base_retriever = VectorRetriever(
                embedding_model=self.embedding_model,
                vector_store=self.vector_store,
                similarity_threshold=0  # rerank会自己控制阈值
            )
            # 创建重排序检索器
            self.retriever = ReRankRetriever(
                base_retriever=base_retriever,
                rerank_model=None,  # 这里需要加载实际的重排序模型
                similarity_threshold=self.config["retriever"]["similarity_threshold"]
            )
        else:  # 默认使用向量检索器
            self.retriever = VectorRetriever(
                embedding_model=self.embedding_model,
                vector_store=self.vector_store,
                similarity_threshold=self.config["retriever"]["similarity_threshold"]
            )

        # 初始化llm_handler
        self.llm_handler = LocalLLMHandler(model_name=self.config["llm"]["model_name"],
                                           max_new_tokens=self.config["llm"]["max_new_tokens"],)

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.config["text_splitter"]["chunk_size"],
                                                            chunk_overlap=self.config["text_splitter"]["chunk_overlap"],
                                                            separators=self.config["text_splitter"]["separators"],)
        logging.info("RAG components initialized successfully")

    def add_documents(self, documents: List[Document],
                      show_progress: bool = True) -> List[str]:
        """添加文档到知识库
        Args:
            documents: 文档列表
            show_progress: 是否显示进度
        Returns:
            文档ID列表
        """
        logger.info(f"Adding {len(documents)} documents to knowledge base")

        all_chunks = []
        all_metadatas = []
        all_doc_ids = []

        for i, doc in enumerate(documents):
            if show_progress and i % 10 == 0:
                logger.info(f"Processing document {i + 1}/{len(documents)}")
            # 分割文本
            chunks = self.text_splitter.split_text(doc.content)

            # 为每个chunk设置metadata
            for j, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata.update({
                    "chunk_id": j,
                    "total_chunks": len(chunks),
                    "doc_id": doc.doc_id,
                })
                all_chunks.append(chunk)
                all_metadatas.append(metadata)
                all_doc_ids.append(f"{doc.doc_id}_chunk_{j}")

        # 2. 批量向量化
        logger.info(f"Embedding {len(all_chunks)} text chunks")
        embeddings = self.embedding_model.embed(all_chunks)  # 内部自动批处理

        # 3. 添加到向量存储
        logger.info(f"Adding embeddings to vector store")
        doc_ids = self.vector_store.add(
            embeddings,
            all_chunks,
            all_metadatas,
            all_doc_ids
        )

        # 4. 保存索引
        self.save_index()

        logger.info(f"Successfully added {len(all_chunks)} chunks to knowledge base")
        return doc_ids


    def add_directory(self, directory_path: str,
                      glob_pattern: str = "**") -> List[str]:
        """添加目录中的所有文档

        Args:
            directory_path: 目录路径
            glob_pattern: 文件匹配模式

        Returns:
            文档ID列表
        """
        loader = DirectoryLoader(directory_path, glob_pattern=glob_pattern)
        documents = loader.load()
        return self.add_documents(documents, show_progress=True)


    def query(self, question: str,
              return_source: bool = True,
              custom_prompt: str = None,
              filter_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """查询RAG系统

        Args:
            question: 用户问题
            return_source: 是否返回源文档
            custom_prompt: 自定义提示词模板
        Returns:
            包含答案和相关信息的字典
        """
        logger.info(f"Processing query: {question}")

        # 检索相关文档
        # retrieved_docs = self.reranker.retrieve(question, k=self.config["retriever"]["top_k"],filter_metadata=filter_metadata)
        retrieved_docs = self.retriever.retrieve(question, k=self.config["retriever"]["top_k"],  filter_metadata=filter_metadata)

        if not retrieved_docs:
            logging.warning("No retrieved documents found ")
            return {
                "answer": "Sorry, I couldn't find any information in the knowledge base to answer your question.",
                "sources":[],
                "retrieved_chunks": [],
            }
        # 准备上下文
        contexts = [doc["text"] for doc in retrieved_docs]

        # 生成答案
        if self.llm_handler:
            if custom_prompt:
                prompt = custom_prompt.format(
                    contexts="\n\n".join(contexts),
                    question=question,
                )
                answer = self.llm_handler.generate(prompt=prompt)
            else:
                answer = self.llm_handler.generate_by_context(question, contexts)

        else:
            # 如果没有llm,返回检索结果
            answer = self._format_retrieval_results(retrieved_docs)

        result = {
            "answer": answer,
            "question": question
        }

        if return_source:
            result["sources"] = [doc["metadata"] for doc in retrieved_docs]
            result["retrieved_chunks"] = retrieved_docs

        return result

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """批量查询

        Args:
            questions: 问题列表

        Returns:
            答案列表
        """
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        return results

    def set_llm_handler(self, llm_handler):
        """设置LLM处理器

        Args:
            llm_handler: LLM处理器实例
        """
        self.llm_handler = llm_handler
        logger.info("LLM handler set successfully")

    def save_index(self):
        """保存向量索引"""
        index_path = self.config["vector_store"]["index_path"]
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(index_path)
        logger.info(f"Index saved to {index_path}")


    def load_index(self, index_path: str = None):
        """加载向量索引
        Args:
            index_path: 索引路径，如果为None则使用配置中的路径
        """
        if index_path is None:
            index_path = self.config["vector_store"]["index_path"]
        self.vector_store.load(index_path)
        logger.info(f"Index loaded from {index_path}")

    def clear_knowledge_base(self):
        """清空知识库"""
        self.vector_store = FaissVectorStore(
            dimension=self.config["vector_store"]["dimension"],
        )
        # 把原来知识库覆盖掉
        self.save_index()
        logging.info("Knowledge base cleared")

    def _format_retrieval_results(self, docs: List[Dict]) -> str:
        """格式化检索结果"""
        formatted = "根据知识库找到以下相关信息：\n\n"

        for i, doc in enumerate(docs[:3], 1):
            formatted += f"{i}. (相似度: {doc['similarity']:.2%})\n"
            formatted += f"   {doc['text'][:200]}...\n\n"

        return formatted

    def set_retriever(self, retriever: Retriever):
        """动态设置检索器"""
        self.retriever = retriever
        logger.info(f"Retriever changed to {type(retriever).__name__}")

    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            "total_documents": self.vector_store.get_stats()["total_vectors"],
            "embedding_model": self.config["embedding"]["model_name"],
            "chunk_size": self.config["text_splitter"]["chunk_size"],
            "retriever_top_k": self.config["retriever"]["top_k"],
        }


class RAGEvaluator:
    """RAG系统评估器"""

    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline

    def evaluate_retrieval(self, test_queries: List[Dict]) -> Dict:
        """
        评估检索质量

        Args:
            test_queries: 测试查询列表，每个包含:
                - question: 查询问题
                - expected_docs: 期望的文档ID列表
                - filter_metadata: 可选的过滤条件

        Returns:
            评估指标
        """
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_mrr = 0  # Mean Reciprocal Rank

        for test_case in test_queries:
            question = test_case["question"]
            expected_docs = set(test_case.get("expected_docs", []))
            filter_metadata = test_case.get("filter_metadata", None)

            # 执行检索
            retrieved = self.rag_pipeline.retriever.retrieve(
                question,
                k=10,  # 获取更多结果以计算MRR
                filter_metadata=filter_metadata
            )

            retrieved_ids = {doc["metadata"].get("doc_id")
                             for doc in retrieved if doc["metadata"].get("doc_id")}

            # 计算指标
            if retrieved_ids and expected_docs:
                intersection = retrieved_ids & expected_docs
                precision = len(intersection) / len(retrieved_ids)
                recall = len(intersection) / len(expected_docs)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                # 计算MRR
                for i, doc in enumerate(retrieved):
                    if doc["metadata"].get("doc_id") in expected_docs:
                        mrr = 1 / (i + 1)
                        break
                else:
                    mrr = 0
            else:
                precision = recall = f1 = mrr = 0

            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_mrr += mrr

        n = len(test_queries)
        return {
            "avg_precision": total_precision / n if n > 0 else 0,
            "avg_recall": total_recall / n if n > 0 else 0,
            "avg_f1": total_f1 / n if n > 0 else 0,
            "avg_mrr": total_mrr / n if n > 0 else 0,
            "num_queries": n
        }



