"""Civis Lucri-Faber API 客户端"""
import os
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class Message:
    """对话消息"""
    role: str  # "system", "user", "assistant"
    content: str


class APIClient:
    """云端 API 客户端

    支持 OpenAI GPT 和 Anthropic Claude
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_anthropic: bool = True,
        model: str = "gpt-4"
    ):
        self.use_anthropic = use_anthropic
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")

        if not self.api_key:
            print("[WARN] No API key set, using mock mode")

        self._client = None

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """发送对话请求"""

        if not self.api_key:
            return self._mock_response(messages)

        if self.use_anthropic:
            return self._anthropic_chat(messages, system_prompt, temperature, max_tokens)
        else:
            return self._openai_chat(messages, system_prompt, temperature, max_tokens)

    def _openai_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """OpenAI API 调用"""
        try:
            from openai import OpenAI
        except ImportError:
            print("[ERROR] Please install openai: pip install openai")
            return self._mock_response(messages)

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        # 构建消息
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] API call failed: {e}")
            return self._mock_response(messages)

    def _anthropic_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Anthropic Claude API 调用"""
        try:
            import anthropic
        except ImportError:
            print("[ERROR] Please install anthropic: pip install anthropic")
            return self._mock_response(messages)

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)

        # 转换消息格式
        claude_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        try:
            response = self._client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=claude_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text
        except Exception as e:
            print(f"[ERROR] API call failed: {e}")
            return self._mock_response(messages)

    def _mock_response(self, messages: List[Dict[str, str]]) -> str:
        """模拟响应 (无 API key 时)"""
        last_msg = messages[-1]["content"] if messages else "hello"
        return f"[MOCK] Received: {last_msg[:50]}..."


def create_api_client(config) -> APIClient:
    """工厂函数: 创建 API 客户端"""
    return APIClient(
        api_key=config.openai_api_key or config.anthropic_api_key,
        use_anthropic=config.use_anthropic,
        model=config.model_name
    )