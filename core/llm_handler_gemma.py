#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : hukangzhe
# @File    : llm_handler_gemma.py
# @Description :
from typing import List
from abc import ABC, abstractmethod
from transformers import pipeline
import torch

class LLMHandler(ABC):
    """LLM处理器基类"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成回复"""
        pass

    @abstractmethod
    def generate_by_context(self, query: str, contexts: List[str], **kwargs) -> str:
        """基于上下文生成回复"""
        pass

class LocalLLMHandler(LLMHandler):
    """本地LLM处理器（使用Hugging Face模型）"""

    def __init__(self, model_name: str = "google/gemma-2-2b-it", max_new_tokens: int = 512):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, prompt: str, **kwargs) -> str:
        """生成回复"""
        pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        messages = [
            {"role": "user", "content": prompt},
        ]
        outputs = pipe(messages, max_new_tokens=self.max_new_tokens, disable_compile=True)
        response = outputs[0]["generated_text"][-1]["content"].strip()
        return response

    def generate_by_context(self, query: str, contexts: List[str], **kwargs) -> str:
        """基于上下文生成回复"""

        context_text = "\n\n".join(contexts)
        prompt = f"""Answer the questions based on the following information:

{context_text}

Question：{query}

Answer:"""

        return self.generate(prompt, **kwargs)


# if __name__ == "__main__":
#     llm = LocalLLMHandler(model_name="google/gemma-2-2b-it", max_new_tokens=256)
#     contexts =[
#         "RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的技术。",
#         "它首先从知识库中检索相关信息，然后基于这些信息生成回答。"
#         "这种方法可以有效解决大语言模型的知识时效性问题和幻觉问题。"
#     ]
#     answer = llm.generate_by_context(query="RAG是什么技术", contexts=contexts)
#     print(answer)