#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/25 10:27
# @Author  : hukangzhe
# @File    : llm_handler_steam.py
# @Description : 流式效果输出测试
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, StoppingCriteria, StoppingCriteriaList
from typing import List, Generator, Tuple
from abc import ABC, abstractmethod
import threading
import queue


class LLMHandler(ABC):
    """ LLM推理基类 """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """生成回复 (修改为返回生成器)"""
        pass

    @abstractmethod
    def generate_by_context(self, query: str, contexts: List[str], **kwargs) -> Generator[str, None, None]:
        """ 根据 上下文生成回复 (修改为返回生成器)"""
        pass


class QwenThinkingStreamer(TextStreamer):
    """
    自定义Streamer，用于分离和流式化Qwen的思考过程和最终答案。
    使用生成器(yield)将内容返回，而不是直接打印。
    """

    def __init__(self, tokenizer: AutoTokenizer, skip_prompt: bool = True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.is_thinking = True
        self.think_end_token_id = 151668  # </think> 对应的 token id
        self.output_queue = queue.Queue()

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """
        这个方法会在每次生成新的、可解码的文本块时被调用。
        我们将在这里实现核心逻辑。
        """
        # TextStreamer的默认实现是直接打印，我们重写它来yield
        # 这里我们将文本放入队列，由外部的生成器消费
        self.output_queue.put(text)
        if stream_end:
            self.output_queue.put(None)  # 发送结束信号

    def __iter__(self):
        return self

    def __next__(self):
        value = self.output_queue.get()
        if value is None:
            raise StopIteration()
        return value

    def generate_output(self) -> Generator[Tuple[str, str], None, None]:
        """
        一个生成器，用于从队列中获取数据并处理思考/回答逻辑
        返回一个元组 (state, content)，state可以是 'thinking' 或 'answer'
        """
        full_decoded_text = ""
        for text_chunk in self:
            if not self.is_thinking:
                # 如果已经结束思考，直接产出回答
                yield "answer", text_chunk
                continue

            # 检查新来的文本块中是否包含结束符
            temp_text = full_decoded_text + text_chunk
            # 使用tokenizer对当前累计的文本进行编码，以准确查找token_id
            # 注意：这是一个简化的检查，在某些边缘情况下可能不完美，但对多数场景有效
            tokens = self.tokenizer.encode(temp_text, add_special_tokens=False)

            if self.think_end_token_id in tokens:
                # 找到了结束符，进行分割
                # 为了精确，我们找到分割点
                split_point = tokens.index(self.think_end_token_id)

                # 解码思考部分
                thinking_part_tokens = tokens[:split_point]
                thinking_text = self.tokenizer.decode(thinking_part_tokens)

                # 解码回答部分
                answer_part_tokens = tokens[split_point + 1:]  # 跳过</think>
                answer_text = self.tokenizer.decode(answer_part_tokens)

                # 产出剩余的思考部分和第一部分回答
                # 我们需要减去已经产出的部分
                remaining_thinking = thinking_text[len(full_decoded_text):]
                if remaining_thinking:
                    yield "thinking", remaining_thinking
                if answer_text:
                    yield "answer", answer_text

                self.is_thinking = False
                full_decoded_text = thinking_text + self.tokenizer.decode(self.think_end_token_id) + answer_text

            else:
                # 还在思考阶段
                yield "thinking", text_chunk
                full_decoded_text += text_chunk



class StreamingLocalLLMHandler(LLMHandler):
    """本地模型回复，支持流式输出"""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", max_new_tokens: int = 256):
        print("Initializing model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Model initialized.")

    def generate(self, prompt: str, **kwargs) -> Generator[Tuple[str, str], None, None]:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 实例化我们自定义的streamer
        streamer = QwenThinkingStreamer(self.tokenizer, skip_prompt=True)

        # generate函数需要在单独的线程中运行，以避免阻塞主线程
        # 主线程则可以从streamer中实时消费内容
        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            streamer=streamer,
            **kwargs
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 从streamer的生成器中产出内容
        yield from streamer.generate_output()

    def generate_by_context(self, query: str, contexts: List[str], **kwargs) -> Generator[Tuple[str, str], None, None]:
        context_text = "\n".join(contexts)
        prompt = f"""Answer the questions based on the following context：
        {context_text}
        Question：{query}
        answer："""

        # 使用 yield from 将 generate 的生成器内容直接传递出去
        yield from self.generate(prompt, **kwargs)


# --- 测试代码 ---
if __name__ == "__main__":
    # 注意：模型较大，首次下载和加载需要时间
    llm = StreamingLocalLLMHandler(model_name="Qwen/Qwen3-0.6B", max_new_tokens=512)
    contexts = [
        "RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的技术。",
        "它首先从知识库中检索相关信息，然后基于这些信息生成回答。",
        "这种方法可以有效解决大语言模型的知识时效性问题和幻觉问题。"
    ]

    print("\n--- 开始流式生成 ---\n")

    current_state = ""
    full_thinking = ""
    full_answer = ""

    # 消费生成器
    for state, content in llm.generate_by_context(query="RAG是什么技术", contexts=contexts):
        if state != current_state:
            current_state = state
            if state == "thinking":
                print("\n[思考过程开始] ...")
            elif state == "answer":
                print("\n\n[回答内容开始] ...")

        print(content, end="", flush=True)

        if state == "thinking":
            full_thinking += content
        elif state == "answer":
            full_answer += content

    print("\n\n--- 流式生成结束 ---\n")
    print("完整的思考过程:\n", full_thinking.strip())
    print("\n完整的回答内容:\n", full_answer.strip())
