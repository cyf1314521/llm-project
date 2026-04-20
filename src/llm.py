import time
import logging
from abc import ABC, abstractmethod

from openai import OpenAI

logger = logging.getLogger(__name__)

# API 调用重试参数：指数退避（2s, 4s, 8s）
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


class BaseLLM(ABC):
    """LLM 抽象接口，便于后续替换不同后端。"""

    @abstractmethod
    def generate(self, messages: list, temperature: float, top_p: float) -> str:
        pass


class ChatLLM(BaseLLM):
    """OpenAI 兼容接口封装，提供统一 generate 方法。"""

    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, messages, temperature=0, top_p=1) -> str:
        """
        调用聊天模型并返回文本结果。
        - 失败时自动重试
        - 超过最大重试次数后返回空字符串
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                )
                return response.choices[0].message.content
            except Exception as e:
                # 指数退避，降低瞬时网络抖动或限流的影响
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "API error (attempt %d/%d): %s. Retrying in %ds...",
                        attempt, MAX_RETRIES, e, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "API error (attempt %d/%d): %s. Giving up.",
                        attempt, MAX_RETRIES, e,
                    )
        return ""