#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/15 11:35
# @Author  : hukangzhe
# @File    : run.py
# @Description :
import os
import sys
import subprocess
from pathlib import Path
import platform
import logging

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]}")


def setup_directories():
    """创建必要的目录"""
    directories = [
        "uploads",
        "exports",
        "data/documents",
        "data/vector_db",
        "cache/embeddings",
        "logs",
        "static/css",
        "static/js",
        "templates"
    ]

    for dir_path in directories:
        path = PROJECT_ROOT / dir_path
        path.mkdir(parents=True, exist_ok=True)

    print("Directory structure created")

def start_server():
    """启动Flask服务器"""
    print("\n" + "=" * 60)
    print("Starting RAG System...")
    print("=" * 60)

    # 设置环境变量
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'

    # 导入并运行Flask应用
    try:
        from app import app

        print("\n" + "=" * 60)
        print("RAG System is ready!")
        print("=" * 60)
        print("\n Access the system at:")
        print("   • Local:    http://localhost:5000")
        print("   • Network:  http://[your-ip]:5000")
        print("\nQuick Start:")
        print("   1. Upload documents (PDF, TXT)")
        print("   2. Ask questions")
        print("   3. Get AI-powered answers")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60 + "\n")

        # 启动服务器
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=True
        )

    except ImportError as e:
        print(f"Failed to import app: {e}")
        print("Please check if app.py exists and has no syntax errors")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)


def main():
    """主函数"""
    print("=" * 60)
    print("🤖 RAG System Setup & Startup")
    print("=" * 60 + "\n")


    check_python_version()


    setup_directories()


    start_server()


if __name__ == "__main__":
    main()
