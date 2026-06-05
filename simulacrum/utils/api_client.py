"""Simulacrum API 客户端

支持 OpenAI GPT、Anthropic Claude、DeepSeek、Ollama 本地模型
"""
import json
import os
from dataclasses import dataclass


@dataclass
class Message:
    """对话消息"""
    role: str  # "system", "user", "assistant"
    content: str


class APIClient:
    """多通道 API 客户端

    支持 OpenAI、Anthropic、DeepSeek（兼容 OpenAI SDK）和 Ollama 本地模型。
    llm_provider 控制优先使用哪个通道：
    - "auto": 按 DeepSeek > OpenAI > Anthropic > Ollama > Mock 顺序尝试
    - 其余: 直接使用指定通道
    """

    def __init__(
        self,
        api_key: str | None = None,
        use_anthropic: bool = True,
        model: str = "gpt-4",
        # 新增多通道参数
        deepseek_api_key: str = "",
        deepseek_model: str = "deepseek-chat",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3",
        llm_provider: str = "auto",
    ):
        self.use_anthropic = use_anthropic
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
        self.deepseek_api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_model = deepseek_model
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.llm_provider = llm_provider

        self._client = None
        self._deepseek_client = None

    def _resolve_provider(self) -> str:
        """auto 模式下自动选择可用的 provider"""
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.deepseek_api_key:
            return "deepseek"
        if self.api_key:
            return "anthropic" if self.use_anthropic else "openai"
        if self._check_ollama():
            return "ollama"
        return "mock"

    def _check_ollama(self) -> bool:
        """检查 Ollama 是否在运行"""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.ollama_base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> str:
        """发送对话请求，自动选择通道

        Args:
            messages: 对话消息列表
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top_p 采样参数
            presence_penalty: 存在惩罚（避免重复内容）
            frequency_penalty: 频率惩罚（降低重复率）
        """

        provider = self._resolve_provider()

        dispatch = {
            "deepseek": self._deepseek_chat,
            "openai": self._openai_chat,
            "anthropic": self._anthropic_chat,
            "ollama": self._ollama_chat,
            "mock": self._mock_response,
        }

        handler = dispatch.get(provider, self._mock_response)
        if provider in ("mock",):
            return handler(messages)

        result = handler(messages, system_prompt, temperature, max_tokens, top_p, presence_penalty, frequency_penalty)
        if result is None:
            return self._mock_response(messages)
        return result

    def _openai_chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> str | None:
        """OpenAI API 调用"""
        try:
            from openai import OpenAI
        except ImportError:
            print("[ERROR] Please install openai: pip install openai")
            return None

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] OpenAI API call failed: {e}")
            return None

    def _anthropic_chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> str | None:
        """Anthropic Claude API 调用"""
        try:
            import anthropic
        except ImportError:
            print("[ERROR] Please install anthropic: pip install anthropic")
            return None

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)

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
                max_tokens=max_tokens,
                top_p=top_p,
            )
            return response.content[0].text
        except Exception as e:
            print(f"[ERROR] Anthropic API call failed: {e}")
            return None

    def _deepseek_chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> str | None:
        """DeepSeek API 调用（兼容 OpenAI SDK，切换 base_url）"""
        try:
            from openai import OpenAI
        except ImportError:
            print("[ERROR] Please install openai: pip install openai")
            return None

        if self._deepseek_client is None:
            self._deepseek_client = OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com"
            )

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        try:
            response = self._deepseek_client.chat.completions.create(
                model=self.deepseek_model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] DeepSeek API call failed: {e}")
            return None

    def _ollama_chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> str | None:
        """Ollama 本地模型调用"""
        try:
            import urllib.request
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)

            payload = json.dumps({
                "model": self.ollama_model,
                "messages": full_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": top_p,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                }
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.ollama_base_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("message", {}).get("content", "")
        except Exception as e:
            print(f"[ERROR] Ollama call failed: {e}")
            return None

    def _mock_response(self, messages: list[dict[str, str]]) -> str:
        """模拟响应 (无 API key 时)"""
        last_msg = messages[-1]["content"] if messages else "hello"
        return f"[MOCK] 我收到了你的消息：{last_msg[:100]}... (当前无 LLM 连接，请配置 API key 或启动 Ollama)"


def create_api_client(config) -> APIClient:
    """工厂函数: 创建 API 客户端"""
    return APIClient(
        api_key=config.openai_api_key or config.anthropic_api_key,
        use_anthropic=config.use_anthropic,
        model=config.model_name,
        deepseek_api_key=getattr(config, 'deepseek_api_key', ''),
        deepseek_model=getattr(config, 'deepseek_model', 'deepseek-chat'),
        ollama_base_url=getattr(config, 'ollama_base_url', 'http://localhost:11434'),
        ollama_model=getattr(config, 'ollama_model', 'llama3'),
        llm_provider=getattr(config, 'llm_provider', 'auto'),
    )
