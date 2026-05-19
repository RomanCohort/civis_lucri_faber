"""Simulacrum 工具注册与执行系统

事件驱动架构下的工具调用系统:
- ToolSpec: 工具描述（名称、参数 schema、处理函数、是否需要授权）
- ToolSystem: 注册/执行/解析工具调用，生成 LLM 可读的工具描述

工具调用协议:
    LLM 回复中包含 [TOOL: name(json_args)] 时触发工具执行，
    结果作为新消息注入对话上下文。
"""
import re
import json
import subprocess
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolSpec:
    """工具定义"""
    name: str
    description: str
    parameters: dict          # JSON Schema 格式
    handler: Callable
    requires_auth: bool = False  # 外部工具需要用户授权


class ToolSystem:
    """工具注册与执行系统"""

    def __init__(self, event_bus=None):
        self._tools: Dict[str, ToolSpec] = {}
        self._bus = event_bus
        self._call_log: List[Dict] = []

    def register(self, tool: ToolSpec) -> None:
        """注册工具"""
        self._tools[tool.name] = tool

    def execute(self, tool_name: str, arguments: dict) -> str:
        """执行工具并返回结果字符串"""
        tool = self._tools.get(tool_name)
        if tool is None:
            return f"[ERROR] Unknown tool: {tool_name}"

        try:
            result = tool.handler(**arguments)
            self._call_log.append({
                "tool": tool_name,
                "args": arguments,
                "result": str(result)[:500],
                "timestamp": datetime.now().isoformat(),
            })
            return str(result)
        except Exception as e:
            return f"[ERROR] Tool '{tool_name}' failed: {e}"

    def parse_tool_call(self, llm_response: str) -> Optional[Tuple[str, dict]]:
        """从 LLM 回复中解析工具调用

        支持格式:
            [TOOL: name({"key": "value"})]
            [TOOL: name(key="value")]
        """
        pattern = r'\[TOOL:\s*(\w+)\s*\((.+?)\)\s*\]'
        match = re.search(pattern, llm_response, re.DOTALL)
        if not match:
            return None

        tool_name = match.group(1)
        args_str = match.group(2).strip()

        try:
            # 尝试 JSON 解析
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            # 尝试 key=value 格式
            arguments = {}
            for pair in args_str.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    k = k.strip().strip('"\'')
                    v = v.strip().strip('"\'')
                    arguments[k] = v
                else:
                    # 位置参数作为 "input"
                    arguments["input"] = pair.strip('"\'')

        return tool_name, arguments

    def get_tools_description(self) -> str:
        """生成给 LLM 的工具描述文本"""
        if not self._tools:
            return ""

        lines = ["\n可用工具（需要时在回复中调用）:"]
        for name, tool in self._tools.items():
            params_desc = ", ".join(
                f'"{k}": {json.dumps(v.get("description", k), ensure_ascii=False)}'
                for k, v in tool.parameters.get("properties", {}).items()
            )
            auth_tag = " [需授权]" if tool.requires_auth else ""
            lines.append(
                f'  - {name}{auth_tag}: {tool.description}。'
                f'参数: {{{params_desc}}}'
            )
        lines.append(
            "\n调用格式: [TOOL: name({\"key\": \"value\"})]"
        )
        return "\n".join(lines)

    def get_call_log(self) -> List[Dict]:
        """获取工具调用历史"""
        return list(self._call_log)

    def get_available_tools(self, bio_state: Dict) -> List[str]:
        """根据生物状态过滤可用工具

        Args:
            bio_state: 包含 bio_state 的字典，如 {'cortisol': 0.8, 'consciousness': 0.3, 'defense': 'freeze'}

        Returns:
            可用工具名称列表
        """
        available = []

        # 1. 意识门控：低意识限制工具
        consciousness = bio_state.get('consciousness', 0.5)
        if consciousness < 0.3:
            # 低意识：只允许基本工具
            for name in self._tools:
                if name in ['get_time', 'memory_recall', 'memory_store']:
                    available.append(name)
            return available

        # 2. 防御行为：冻结时禁用外部工具
        defense = bio_state.get('defense', '')
        if defense == 'freeze':
            # 冻结：只允许最基本工具
            for name in self._tools:
                if name in ['get_time', 'memory_recall']:
                    available.append(name)
            return available

        # 3. 高压力（皮质醇>0.7）：禁用探索性工具
        cortisol = bio_state.get('cortisol', 0.3)
        if cortisol > 0.7:
            # 高压力：禁用搜索、网络等探索工具
            for name, tool in self._tools.items():
                if tool.requires_auth:
                    # 外部工具需要授权，高压力下禁用
                    continue
                if name not in ['get_time', 'memory_recall', 'memory_store']:
                    available.append(name)
            return available

        # 4. 中等压力（皮质醇>0.4）：限制部分工具
        if cortisol > 0.4:
            # 中等压力：禁用高风险工具
            for name, tool in self._tools.items():
                if tool.requires_auth:
                    continue
                if name not in ['get_time', 'memory_recall', 'memory_store', 'memory_store']:
                    available.append(name)
            return available

        # 5. 低警觉度（警觉度<0.3）：限制工具使用频率
        alertness = bio_state.get('alertness', 0.5)
        if alertness < 0.3:
            # 低警觉：只允许基本工具
            for name in self._tools:
                if name in ['get_time', 'memory_recall', 'memory_store']:
                    available.append(name)
            return available

        # 6. 正常状态：允许所有工具
        return list(self._tools.keys())


# ====================================================================
# 默认工具实现
# ====================================================================

def make_get_time(agent_ref):
    """获取当前时间 + agent 主观昼夜节律"""
    def handler():
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        circadian_hour = agent_ref._internal_state.get('scn_circadian_hour', 12.0)
        melatonin = agent_ref._internal_state.get('scn_melatonin', 0.3)
        alertness = agent_ref._internal_state.get('scn_alertness', 0.5)
        return (
            f"客观时间: {now}\n"
            f"主观昼夜: {circadian_hour:.1f}时\n"
            f"褪黑素: {melatonin:.4f}\n"
            f"警觉度: {alertness:.4f}"
        )
    return handler


def make_memory_recall(agent_ref):
    """查询海马体/知识库"""
    def handler(query: str = "", top_k: int = 5):
        results = []
        # 优先查询海马体
        if query:
            try:
                import numpy as np
                query_vec = np.full(64, 0.5, dtype=np.float32)
                # 简单用 hash 编码 query 到向量
                for i, ch in enumerate(query[:64]):
                    query_vec[i % 64] = (ord(ch) % 100) / 100.0
                episodes = agent_ref.hippocampus.retrieve(query_vec, top_k=top_k)
                for ep in episodes:
                    results.append(f"[海马体] action={ep.action}, reward={ep.reward:.3f}")
            except Exception as e:
                results.append(f"[海马体查询失败: {e}]")

        # 查询知识库
        try:
            memories = agent_ref.memory.get_recent_memories(n=top_k)
            for mem in memories:
                results.append(f"[知识库] {mem.content[:100]} (重要性: {mem.importance:.2f})")
        except Exception:
            pass

        if not results:
            return "没有找到相关记忆。"
        return "\n".join(results)
    return handler


def make_memory_store(agent_ref):
    """存储知识到知识库"""
    def handler(content: str, importance: float = 0.5, tags: str = ""):
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else ["user_input"]
        try:
            agent_ref.memory.add_memory(
                content=content,
                importance=importance,
                tags=tag_list,
            )
            return f"已存储: {content[:50]}... (重要性: {importance})"
        except Exception as e:
            return f"存储失败: {e}"
    return handler


def make_get_internal_state(agent_ref):
    """自省：读取内部生理/情绪指标"""
    def handler():
        s = agent_ref._internal_state
        keys = [
            'bsm_arousal', 'bsm_arousal_name', 'bsm_heart_rate', 'bsm_respiratory_rate',
            'bsm_blood_pressure', 'bsm_pain_gating',
            'nt_dopamine', 'nt_serotonin', 'nt_state',
            'cortisol_level', 'allostatic_load',
            'scn_circadian_hour', 'scn_melatonin', 'scn_alertness',
            'scn_sleep_pressure', 'scn_wake_drive',
            'limbic_emotion', 'limbic_valence', 'limbic_arousal',
            'mood_valence', 'mood_arousal',
            'sleep_fatigue', 'sleep_stage',
            'pfc_maturity', 'pfc_inhibition',
            'brain_health', 'brain_waste', 'neuroinflammation', 'myelination_level',
            'hormone_oxytocin', 'hormone_adrenaline', 'hormone_cortisol',
            'ans_hrv', 'ans_polyvagal_state',
            'encoding_modulation', 'plasticity_bdnf',
        ]
        lines = ["=== 内部生理状态 ==="]
        for k in keys:
            v = s.get(k, "N/A")
            if isinstance(v, float):
                v = f"{v:.4f}"
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    return handler


def make_shell_exec():
    """执行 shell 命令"""
    def handler(command: str, timeout: int = 30):
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR] {result.stderr}"
            if not output.strip():
                output = "(无输出)"
            return output[:2000]  # 截断防止过长
        except subprocess.TimeoutExpired:
            return f"[ERROR] 命令超时 ({timeout}s)"
        except Exception as e:
            return f"[ERROR] {e}"
    return handler


def make_file_read():
    """读取文件"""
    def handler(path: str, max_lines: int = 100):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:max_lines]
            content = "".join(lines)
            return content[:5000] if len(content) > 5000 else content
        except FileNotFoundError:
            return f"[ERROR] 文件不存在: {path}"
        except Exception as e:
            return f"[ERROR] {e}"
    return handler


def make_file_write():
    """写入文件"""
    def handler(path: str, content: str, append: bool = False):
        try:
            mode = 'a' if append else 'w'
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)
            return f"已写入: {path} ({len(content)} 字符)"
        except Exception as e:
            return f"[ERROR] {e}"
    return handler


def register_default_tools(tool_system: ToolSystem, agent_ref) -> None:
    """注册所有默认工具"""
    # 内部工具（无需授权）
    tool_system.register(ToolSpec(
        name="get_time",
        description="获取当前客观时间和 agent 主观昼夜节律",
        parameters={"type": "object", "properties": {}},
        handler=make_get_time(agent_ref),
        requires_auth=False,
    ))
    tool_system.register(ToolSpec(
        name="memory_recall",
        description="从海马体和知识库中检索相关记忆",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词"},
                "top_k": {"type": "integer", "description": "返回数量，默认5"},
            },
        },
        handler=make_memory_recall(agent_ref),
        requires_auth=False,
    ))
    tool_system.register(ToolSpec(
        name="memory_store",
        description="将新知识存储到知识库",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要存储的内容"},
                "importance": {"type": "number", "description": "重要性 0-1，默认0.5"},
                "tags": {"type": "string", "description": "标签，逗号分隔"},
            },
        },
        handler=make_memory_store(agent_ref),
        requires_auth=False,
    ))
    tool_system.register(ToolSpec(
        name="get_internal_state",
        description="自省：读取 agent 内部生理/情绪/认知指标",
        parameters={"type": "object", "properties": {}},
        handler=make_get_internal_state(agent_ref),
        requires_auth=False,
    ))

    # 外部工具（需授权）
    tool_system.register(ToolSpec(
        name="shell_exec",
        description="执行 shell 命令（需授权）",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的命令"},
                "timeout": {"type": "integer", "description": "超时秒数，默认30"},
            },
        },
        handler=make_shell_exec(),
        requires_auth=True,
    ))
    tool_system.register(ToolSpec(
        name="file_read",
        description="读取文件内容",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "max_lines": {"type": "integer", "description": "最大行数，默认100"},
            },
        },
        handler=make_file_read(),
        requires_auth=True,
    ))
    tool_system.register(ToolSpec(
        name="file_write",
        description="写入文件",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "写入内容"},
                "append": {"type": "boolean", "description": "是否追加模式，默认否"},
            },
        },
        handler=make_file_write(),
        requires_auth=True,
    ))
