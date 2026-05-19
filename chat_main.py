"""Civis Lucri-Faber 交互式聊天入口

运行示例:
    # 使用环境变量中的 API key (DeepSeek / OpenAI / Anthropic)
    python chat_main.py

    # 指定 provider
    python chat_main.py --provider deepseek
    python chat_main.py --provider ollama
    python chat_main.py --provider mock

    # 指定 Ollama 模型
    python chat_main.py --provider ollama --ollama-model qwen2.5:7b
"""
import sys
import os
import argparse

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from civis_lucri_faber.utils.config import Config, load_config
from civis_lucri_faber.core.agent import CivisLucriFaber


EMOJI_MAP = {
    "joy": "\U0001f60a",
    "sadness": "\U0001f622",
    "anger": "\U0001f620",
    "fear": "\U0001f628",
    "neutral": "\U0001f610",
}


def main():
    parser = argparse.ArgumentParser(description="Civis Lucri-Faber Chat")
    parser.add_argument("--provider", default="auto",
                        choices=["auto", "openai", "anthropic", "deepseek", "ollama", "mock"],
                        help="LLM provider")
    parser.add_argument("--ollama-model", default="llama3", help="Ollama 模型名")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama 地址")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # 加载配置
    config = load_config(
        llm_provider=args.provider,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_url,
        chat_temperature=args.temperature,
        chat_max_tokens=args.max_tokens,
        device=args.device,
        initial_balance=100.0,
    )

    print("=" * 60)
    print("  Civis Lucri-Faber - Bio-Inspired Chat Agent")
    print("=" * 60)
    print(f"  LLM Provider: {config.llm_provider}")
    print(f"  Device: {config.device}")
    print(f"  输入 quit / exit 退出")
    print("=" * 60)

    # 实例化 agent
    print("\n[INIT] 初始化仿生大脑...")
    agent = CivisLucriFaber(config=config)
    print("[OK] 初始化完成\n")

    # 交互循环
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[BYE]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("[BYE]")
            break

        try:
            response = agent.chat(user_input)
        except Exception as e:
            print(f"[ERROR] {e}")
            continue

        emoji = EMOJI_MAP.get(response.emotion, "\U0001f610")
        # 状态栏：情绪 + 策略 + 质量 + LLM参数 + 学习
        lp = response.llm_params or {}
        cg = response.cognitive_gate or {}
        qf = response.quality_filter or {}
        strategy = cg.get('strategy', '?')
        verdict = qf.get('verdict', '?')
        learning_tag = " [学习]" if response.learning_active else ""

        state_line = (
            f"  [{emoji} {response.emotion} | "
            f"策略={strategy} | 质量={verdict}"
            f"{learning_tag} | "
            f"T={lp.get('temperature', 0.7):.2f}]"
        )

        print(f"\n{state_line}")
        print(f"CLF: {response.text}")

        # 如果有工具调用，显示
        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  [TOOL] {tc['tool']}({tc['args']})")


if __name__ == "__main__":
    main()
