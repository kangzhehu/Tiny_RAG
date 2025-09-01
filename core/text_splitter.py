#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/21 13:27
# @Author  : hukangzhe
# @File    : text_splitter.py
# @Description : 文本分割，将长文段分成小块
from typing import List
import re


class TextSplitter:
    """文本分割 基类"""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError


class RecursiveCharacterTextSplitter(TextSplitter):
    """递归字符文本分割器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50,
                 separators: List[str] = None):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", "；", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """递归分割文本"""
        chunks = self._recursive_split(text, self.separators)
        # 合并小块，确保满足chunk_size
        merged_chunks = self._merge_chunks(chunks)
        return merged_chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """递归分割文本"""
        if not text:
            return []

        # 如果文本已经小于chunk_size，直接返回
        if len(text) <= self.chunk_size:
            return [text]

        # 尝试使用当前分隔符列表
        for i, separator in enumerate(separators):
            if separator == "":
                # 最后的手段：按字符分割
                return self._split_by_character(text)

            if separator in text:
                # 使用当前分隔符分割，并传入剩余的分隔符列表
                return self._split_by_separator_recursive(
                    text,
                    separator,
                    separators[i + 1:] if i + 1 < len(separators) else [""]
                )

        # 如果没有找到任何分隔符，按字符分割
        return self._split_by_character(text)

    def _split_by_separator_recursive(self, text: str, separator: str,
                                      remaining_separators: List[str]) -> List[str]:
        """递归地按分隔符分割"""
        parts = text.split(separator)
        chunks = []
        current_chunk = ""

        for part in parts:
            # 如果part本身就超过chunk_size，需要进一步分割
            if len(part) > self.chunk_size:
                # 先保存当前累积的chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # 递归分割超大的part
                sub_chunks = self._recursive_split(part, remaining_separators)
                chunks.extend(sub_chunks)

            elif len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                # 可以添加到当前chunk
                if current_chunk:
                    current_chunk += separator + part
                else:
                    current_chunk = part

            else:
                # 当前chunk已满，保存并开始新的chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_character(self, text: str) -> List[str]:
        """按字符分割"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def _merge_chunks(self, chunks: List[str]) -> List[str]:
        """合并小块并添加重叠"""
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for chunk in chunks[1:]:
            if len(current) + len(chunk) + 1 <= self.chunk_size:  # +1 for space
                current += " " + chunk
            else:
                merged.append(current)
                # 添加重叠部分
                if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                    overlap_start = len(current) - self.chunk_overlap
                    current = current[overlap_start:] + " " + chunk
                else:
                    current = chunk

        if current:
            merged.append(current)
        return merged


class RecursiveCharacterTextSplitter2(TextSplitter):
    """重叠后的chunk不超过chunk_size"""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50,
                 separators: List[str] = None):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        # 使用split_size来控制初始分割
        self.split_size = chunk_size - chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """主分割方法"""
        # 步骤1：按split_size分割
        chunks = self._recursive_split(text, self.separators)

        # 步骤2：添加重叠（这会让最终chunk大小接近chunk_size）
        chunks_with_overlap = self._add_overlap(chunks)

        return chunks_with_overlap

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """递归分割，使用split_size"""
        if not text:
            return []

        # 如果文本已经小于split_size，直接返回
        if len(text) <= self.split_size:
            return [text]

        for i, separator in enumerate(separators):
            if separator == "":
                return self._split_by_character(text)

            if separator in text:
                return self._split_by_separator_recursive(
                    text,
                    separator,
                    separators[i + 1:] if i + 1 < len(separators) else [""]
                )

        return self._split_by_character(text)

    def _split_by_separator_recursive(self, text: str, separator: str,
                                      remaining_separators: List[str]) -> List[str]:
        """使用split_size进行分割"""
        parts = text.split(separator)
        chunks = []
        current_chunk = ""

        for part in parts:
            # 关键改动：使用split_size
            if len(part) > self.split_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                sub_chunks = self._recursive_split(part, remaining_separators)
                chunks.extend(sub_chunks)

            elif len(current_chunk) + len(part) + len(separator) <= self.split_size:
                if current_chunk:
                    current_chunk += separator + part
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _split_by_character(self, text: str) -> List[str]:
        """按字符分割，使用split_size"""
        chunks = []
        for i in range(0, len(text), self.split_size):
            chunk = text[i:i + self.split_size]
            chunks.append(chunk)
        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """添加重叠部分"""
        if not chunks or len(chunks) == 1 or self.chunk_overlap == 0:
            return chunks

        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]  # 注意：使用原始chunks的前一个
            curr_chunk = chunks[i]

            # 计算重叠大小
            overlap_size = min(self.chunk_overlap, len(prev_chunk))

            if overlap_size > 0:
                overlap_text = prev_chunk[-overlap_size:]
                new_chunk = overlap_text + curr_chunk
                # 最终的chunk大小约为 split_size + chunk_overlap = chunk_size
                overlapped.append(new_chunk)
            else:
                overlapped.append(curr_chunk)

        return overlapped


class SentenceSplitter(TextSplitter):
    """句子分割器"""
    def split_text(self, text: str) -> List[str]:
        sentences = re.split(r'[。！？；\n]+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk)+len(sentence)<=self.chunk_size:
                current_chunk += sentence+"。"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence+" "    # 新的缓存chunk

        if current_chunk:
            chunks.append(current_chunk)
        final_chunks = self._add_overlap(chunks)
        return final_chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        merged = []
        merged.append(chunks[0])
        for i in range(1, len(chunks)):
            per_chunk = merged[-1]
            cur_chunk = chunks[i]
            overpart = per_chunk[-self.chunk_overlap:]
            merged.append(overpart+cur_chunk)

        return merged

