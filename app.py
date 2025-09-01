#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/4 11:35
# @Author  : hukangzhe
# @File    : app.py
# @Description :
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
import logging
import time

from core.rag_pipeline import RAGPipeline, RAGEvaluator
from core.document_loader import PDFLoader, TextLoader, Document
from core.retriever import HybridRetriever, VectorRetriever, EnsembleRetriever
from core.utils import PerformanceMonitor, save_json, load_json, count_tokens, clean_text

# 配置
app = Flask(__name__)
CORS(app)

# 文件上传配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('data/vector_db', exist_ok=True)
os.makedirs('exports', exist_ok=True)

# 初始化RAG系统
rag_pipeline = RAGPipeline(config_path="config.yaml")
performance_monitor = PerformanceMonitor()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 存储会话历史和文档信息
session_history = {}
uploaded_documents = {}  # 存储上传的文档信息


def allowed_file(filename):
    """检查文件类型"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """获取系统配置"""
    try:
        config = {
            "retriever_types": ["vector", "hybrid", "ensemble"],
            "current_retriever": type(rag_pipeline.retriever).__name__,
            "embedding_model": rag_pipeline.config["embedding"]["model_name"],
            "chunk_size": rag_pipeline.config["text_splitter"]["chunk_size"],
            "chunk_overlap": rag_pipeline.config["text_splitter"]["chunk_overlap"],
            "similarity_threshold": rag_pipeline.config["retriever"]["similarity_threshold"],
            "top_k": rag_pipeline.config["retriever"]["top_k"]
        }

        return jsonify({
            'success': True,
            'config': config
        })
    except Exception as e:
        logger.error(f"Config error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/retriever', methods=['POST'])
def update_retriever():
    """更新检索器配置"""
    try:
        data = request.json
        retriever_type = data.get('type', 'vector')
        params = data.get('params', {})

        # 根据类型创建新的检索器
        if retriever_type == 'hybrid':
            new_retriever = HybridRetriever(
                rag_pipeline.embedding_model,
                rag_pipeline.vector_store,
                keyword_weight=params.get('keyword_weight', 0.3),
                similarity_threshold=params.get('similarity_threshold', 0.5)
            )
        elif retriever_type == 'ensemble':
            retrievers = [
                VectorRetriever(rag_pipeline.embedding_model, rag_pipeline.vector_store),
                HybridRetriever(rag_pipeline.embedding_model, rag_pipeline.vector_store)
            ]
            weights = params.get('weights', [0.6, 0.4])
            new_retriever = EnsembleRetriever(retrievers, weights)
        else:  # vector
            new_retriever = VectorRetriever(
                rag_pipeline.embedding_model,
                rag_pipeline.vector_store,
                similarity_threshold=params.get('similarity_threshold', 0.5)
            )

        # 更新检索器
        rag_pipeline.set_retriever(new_retriever)

        # 更新配置
        rag_pipeline.config["retriever"]["type"] = retriever_type
        for key, value in params.items():
            rag_pipeline.config["retriever"][key] = value

        return jsonify({
            'success': True,
            'message': f'Retriever updated to {retriever_type}',
            'config': params
        })

    except Exception as e:
        logger.error(f"Update retriever error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """文件上传接口"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        files = request.files.getlist('file')
        preprocess = request.form.get('preprocess', 'true').lower() == 'true'

        uploaded_files = []
        all_documents = []

        for file in files:
            if file.filename == '':
                continue

            if file and allowed_file(file.filename):
                # 安全的文件名
                filename = secure_filename(file.filename)
                file_id = str(uuid.uuid4())

                # 保存文件
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
                file.save(file_path)

                # 加载文档
                if filename.endswith('.pdf'):
                    loader = PDFLoader(file_path)
                elif filename.endswith('.txt'):
                    loader = TextLoader(file_path)
                else:
                    continue

                documents = loader.load()

                # 预处理文档（如果启用）
                if preprocess:
                    for doc in documents:
                        # 清理文本
                        doc.content = clean_text(doc.content)
                        # 添加token计数
                        doc.metadata['token_count'] = count_tokens(doc.content)

                all_documents.extend(documents)

                # 记录文档信息
                doc_info = {
                    'filename': filename,
                    'file_id': file_id,
                    'doc_count': len(documents),
                    'upload_time': datetime.now().isoformat(),
                    'preprocessed': preprocess
                }

                uploaded_documents[file_id] = doc_info
                uploaded_files.append(doc_info)

                logger.info(f"Successfully uploaded: {filename}")

        # 添加到RAG系统
        if all_documents:
            # 记录性能
            start_time = time.time()
            doc_ids = rag_pipeline.add_documents(all_documents)
            process_time = time.time() - start_time

            # 保存索引
            rag_pipeline.save_index()

            # 更新文档信息
            for file_info in uploaded_files:
                file_info['chunk_count'] = len([
                    doc_id for doc_id in doc_ids
                    if doc_id.startswith(file_info['file_id'])
                ])
                file_info['process_time'] = process_time

        return jsonify({
            'success': True,
            'files': uploaded_files,
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'total_chunks': len(doc_ids) if all_documents else 0,
            'process_time': process_time if all_documents else 0
        })

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def query():
    """查询接口"""
    try:
        data = request.json
        question = data.get('question', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        use_history = data.get('use_history', False)
        filter_metadata = data.get('filter_metadata', None)
        return_debug = data.get('return_debug', False)

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        # 获取会话历史
        if use_history and session_id in session_history:
            history = session_history[session_id]
            # 可以将历史信息加入到查询中
        else:
            history = []

        # 执行查询并记录性能
        start_time = time.time()

        # 检索阶段
        retrieval_start = time.time()
        result = rag_pipeline.query(
            question,
            return_source=True,
            filter_metadata=filter_metadata
        )
        retrieval_time = time.time() - retrieval_start

        total_time = time.time() - start_time

        # 记录性能
        performance_monitor.record_query(
            total_time,
            retrieval_time,
            total_time - retrieval_time  # 生成时间
        )

        # 保存到历史
        if session_id not in session_history:
            session_history[session_id] = []

        session_history[session_id].append({
            'question': question,
            'answer': result['answer'],
            'timestamp': datetime.now().isoformat(),
            'performance': {
                'total_time': total_time,
                'retrieval_time': retrieval_time
            }
        })

        # 限制历史长度
        if len(session_history[session_id]) > 10:
            session_history[session_id].pop(0)

        # 准备响应
        response = {
            'success': True,
            'answer': result['answer'],
            'question': question,
            'session_id': session_id,
            'sources': [],
            'performance': {
                'total_time': round(total_time, 3),
                'retrieval_time': round(retrieval_time, 3),
                'generation_time': round(total_time - retrieval_time, 3)
            }
        }

        # 添加来源信息
        if 'retrieved_chunks' in result:
            for chunk in result['retrieved_chunks'][:5]:  # 最多返回5个
                source_info = {
                    'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                    'similarity': float(chunk['similarity']),
                    'metadata': chunk.get('metadata', {}),
                    'retriever_type': chunk.get('retriever_type', 'unknown')
                }

                # 添加额外的分数信息
                if 'hybrid_score' in chunk:
                    source_info['hybrid_score'] = float(chunk['hybrid_score'])
                if 'keyword_score' in chunk:
                    source_info['keyword_score'] = float(chunk['keyword_score'])
                if 'ensemble_score' in chunk:
                    source_info['ensemble_score'] = float(chunk['ensemble_score'])

                response['sources'].append(source_info)

        # 调试信息
        if return_debug:
            response['debug'] = {
                'retriever_type': type(rag_pipeline.retriever).__name__,
                'num_chunks_searched': rag_pipeline.vector_store.get_stats()['total_vectors'],
                'filter_applied': filter_metadata is not None
            }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """评估系统性能"""
    try:
        data = request.json
        test_queries = data.get('test_queries', [])

        if not test_queries:
            return jsonify({'error': 'Test queries required'}), 400

        evaluator = RAGEvaluator(rag_pipeline)

        # 执行评估
        metrics = evaluator.evaluate_retrieval(test_queries)

        return jsonify({
            'success': True,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """列出所有文档"""
    try:
        return jsonify({
            'success': True,
            'documents': list(uploaded_documents.values()),
            'total': len(uploaded_documents)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/<file_id>', methods=['DELETE'])
def delete_document(file_id):
    """删除特定文档"""
    try:
        if file_id in uploaded_documents:
            # 从向量库中删除
            doc_ids_to_delete = [
                doc_id for doc_id in rag_pipeline.vector_store.doc_ids
                if doc_id.startswith(file_id)
            ]

            if doc_ids_to_delete:
                rag_pipeline.vector_store.delete(doc_ids_to_delete)
                rag_pipeline.save_index()

            # 删除文件
            doc_info = uploaded_documents[file_id]
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                f"{file_id}_{doc_info['filename']}"
            )
            if os.path.exists(file_path):
                os.remove(file_path)

            # 从记录中删除
            del uploaded_documents[file_id]

            return jsonify({
                'success': True,
                'message': f'Document {file_id} deleted'
            })
        else:
            return jsonify({'error': 'Document not found'}), 404

    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear', methods=['POST'])
def clear_knowledge_base():
    """清空知识库"""
    try:
        rag_pipeline.clear_knowledge_base()

        # 清空上传文件夹
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        # 清空记录
        uploaded_documents.clear()
        session_history.clear()

        return jsonify({
            'success': True,
            'message': 'Knowledge base cleared successfully'
        })

    except Exception as e:
        logger.error(f"Clear error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取系统统计信息"""
    try:
        # 基础统计
        stats = rag_pipeline.get_stats()

        # 性能统计
        perf_stats = performance_monitor.get_stats()

        # 缓存统计
        cache_stats = rag_pipeline.embedding_model.get_cache_stats()

        # 存储统计
        upload_folder_size = sum(
            os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], f))
            for f in os.listdir(app.config['UPLOAD_FOLDER'])
            if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))
        ) / (1024 * 1024)  # MB

        return jsonify({
            'success': True,
            'stats': {
                **stats,
                'upload_folder_size': upload_folder_size,
                'session_count': len(session_history),
                'document_count': len(uploaded_documents),
                'cache_files': cache_stats['cache_files'],
                'cache_size_mb': cache_stats['cache_size_mb'],
                **perf_stats
            }
        })

    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export', methods=['GET'])
def export_knowledge_base():
    """导出知识库"""
    try:
        from zipfile import ZipFile
        import tempfile

        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with ZipFile(tmp_file.name, 'w') as zipf:
                # 导出配置
                config_data = {
                    'config': rag_pipeline.config,
                    'stats': rag_pipeline.get_stats(),
                    'documents': list(uploaded_documents.values()),
                    'performance': performance_monitor.get_stats()
                }

                # 保存配置
                config_path = os.path.join(tempfile.gettempdir(), 'config.json')
                save_json(config_data, config_path)
                zipf.write(config_path, 'config.json')

                # 添加索引文件
                index_path = rag_pipeline.config["vector_store"]["index_path"]
                if os.path.exists(f"{index_path}.index"):
                    zipf.write(f"{index_path}.index", 'index.faiss')
                if os.path.exists(f"{index_path}.meta"):
                    zipf.write(f"{index_path}.meta", 'index.meta')

                # 添加上传的文档
                for file in os.listdir(app.config['UPLOAD_FOLDER']):
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                    if os.path.isfile(file_path):
                        zipf.write(file_path, f"documents/{file}")

            # 返回文件
            return send_from_directory(
                tempfile.gettempdir(),
                os.path.basename(tmp_file.name),
                as_attachment=True,
                download_name=f"rag_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            )

    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)