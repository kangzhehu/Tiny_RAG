#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : hukangzhe
# @File    : llm_handler.py
# @Description : LLM推理(Qwen大模型)
import time
from abc import ABC, abstractmethod
from typing import List
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import openai
import torch


class LLMHandler(ABC):
    """ LLM推理基类 """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成回复"""
        pass

    @abstractmethod
    def generate_by_context(self, query: str, contexts: List[str], **kwargs) -> str:
        """ 根据 上下文生成回复 """
        pass


class OpenAIHandler(LLMHandler):
    """ 网络API 处理"""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.7, max_tokens: int = 1000,
                 max_retries: int = 3):
        self.client = openai.Client(api_key=api_key, max_retries=max_retries)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def generate(self, prompt: str, **kwargs) -> str:
        """调用OpenAI API生成回复"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # 合并参数
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }

        # 重试机制
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise e

    def generate_by_context(self, query: str, contexts: List[str], system_prompt: str = None, **kwargs) -> str:
        """ 基于上下文 回答 """
        # 构建提示词
        if not system_prompt:
            system_prompt = """You are a helpful assistant. 
                    Answer the question based on the given context. 
                    If the context doesn't contain relevant information, say so."""

        # 组合上下文
        context_text = "\n\n".join([f"Context {i + 1}: {ctx}"
                                    for i, ctx in enumerate(contexts)])

        # 构建完整的prompt
        prompt = f"""Context information:
        {context_text}

        Question: {query}

        Please provide a comprehensive answer based on the above context."""

        messages = [
            {"role": "system", "context": system_prompt},
            {"role": "user", "context": prompt}
        ]
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content


class LocalLLMHandler(LLMHandler):
    """本地模型回复"""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", max_new_tokens: int=256):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        # gemma大模型不同，所以使用和prompt有所不同
        messages = [
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return thinking_content + " "+content

    def generate_by_context(self, query: str, contexts: List[str], **kwargs) -> str:
        """基于上下文生成回复"""

        context_text = "\n".join(contexts)
        prompt = f"""Answer the questions based on the following context：
        {context_text}
        Question：{query}
        answer："""

        return self.generate(prompt, **kwargs)


class PromptTemplate:
    """提示词模板"""

    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        """格式化提示词"""
        return self.template.format(**kwargs)

    @classmethod
    def from_template(cls, template: str):
        """从模板字符串创建"""
        import re
        # 提取变量名
        variables = re.findall(r'\{(\w+)\}', template)
        return cls(template, variables)


# 预定义提示词模板
class PromptTemplates:
    """常用提示词模板集合"""

    QA_TEMPLATE = PromptTemplate.from_template("""Answer the question based on the following context.
    If there is no relevant information in the context, please state that the answer cannot be given.

    上下文信息：
    {context}
    
    Question：{question}
    
    answer：""")

    SUMMARY_TEMPLATE = PromptTemplate.from_template("""Please summarize the main content of the following text：

    {text}
    
    summary：""")

    EXTRACT_TEMPLATE = PromptTemplate.from_template("""Extract {entity_type} from the following text:

    {text}

    Extraction result:""")

# if __name__ == "__main__":
#     llm = LocalLLMHandler(model_name="Qwen/Qwen3-0.6B", max_new_tokens=256)
#     contexts =[
#         "RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的技术。",
#         "它首先从知识库中检索相关信息，然后基于这些信息生成回答。"
#         "这种方法可以有效解决大语言模型的知识时效性问题和幻觉问题。"
#     ]
#     answer = llm.generate_by_context(query="RAG是什么技术", contexts=contexts)
#     print(answer)