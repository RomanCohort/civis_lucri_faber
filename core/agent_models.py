"""Agent data models — AgentState and ChatResponse dataclasses

These dataclasses are separated from agent.py to avoid circular imports
when mixin modules need to reference them.
"""

from dataclasses import dataclass


@dataclass
class AgentState:
    """智能体状态"""
    step: int
    status: str  # "ACTIVE", "HIBERNATE", "DEAD"
    balance: float
    current_goal: str | None
    info_gain: float
    alignment_score: float


@dataclass
class ChatResponse:
    """对话响应"""
    text: str
    emotion: str
    arousal: float
    valence: float
    internal_state: dict
    tool_calls: list[dict]
    llm_params: dict[str, float] = None
    cognitive_gate: dict = None     # Pre-LLM 门控结果
    quality_filter: dict = None     # Post-LLM 过滤结果
    learning_active: bool = False   # 是否触发了主动学习
    vocalization: dict = None       # 发音系统输出（共振峰/声学特征）
    censor_result: dict = None      # Censor 微表情分析结果