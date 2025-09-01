#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/20 9:51
# @Author  : hukangzhe
# @File    : document_loader.py
# @Description :
import glob
import os

import PyPDF2
import chardet
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Document:
    content: str  # 内容
    metadata: Dict  # source page type etc
    doc_id: str


class DocumentLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        raise NotImplementedError


class TextLoader(DocumentLoader):
    """ Txt文件加载器 """
    def load(self) -> List[Document]:
        # 检查文件编码
        with open(self.file_path, 'rb') as file:
            data = file.read()
            encoding = chardet.detect(data)['encoding']
        # 读取内容
        with open(self.file_path, 'r', encoding=encoding) as file:
            context = file.read()

        return [Document(
            content=context,
            metadata={
                "source": self.file_path,
                "type": "text"
            },
            doc_id=f"doc_{hash(self.file_path)}"
        )]


class PDFLoader(DocumentLoader):
    """ PDF文件加载器 """
    def load(self) -> List[Document]:
        documents = []
        with open(self.file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages):
                context = page.extract_text()
                documents.append(Document(
                    content=context,
                    metadata={
                        "source": self.file_path,
                        "type": "PDF",
                        "page": page_num+1
                    },
                    doc_id=f"doc_{hash(self.file_path)}_{page_num}"
                ))
        return documents


class DirectoryLoader:
    """
    目录文件加载器
    """
    def __init__(self, directory: str, glob_pattern: str = "**"):
        self.directory_path = directory
        self.glob_pattern = glob_pattern

    def load(self) -> List[Document]:
        documents = []
        # 遍历整个
        # for root, dirs, files in os.walk(self.directory_path):
        #     for file in files:
        #         file_path = os.path.join(root, file)
        #         if file.endswith('.txt'):
        #             loader = TextLoader(file_path)
        #         elif file.endswith('.pdf'):
        #             loader = PDFLoader(file_path)
        #         else:
        #             continue
        #
        #         documents.extend(loader.load())

        # 用glob进行匹配
        file_paths = glob.glob(os.path.join(self.directory_path, self.glob_pattern), recursive=True)
        for file_path in file_paths:
            if os.path.isfile(file_path):  # 确保是文件而非目录
                if file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                elif file_path.endswith('.pdf'):
                    loader = PDFLoader(file_path)
                else:
                    continue

                documents.extend(loader.load())

        return documents





