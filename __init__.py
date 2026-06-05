"""Simulacrum - 生物启发式 AI 智能体系统"""

# Lazy imports to avoid loading heavy dependencies during testing
# The main classes are available via lazy loading when needed
__version__ = "0.1.0"

def __getattr__(name: str):
    """Lazy import for main classes to avoid import chain issues."""
    if name == "Simulacrum" or name == "AgentState":
        # Handle naming conflict: both core/agent.py and core/agent/ package exist
        import importlib.util
        import sys
        from pathlib import Path
        _agent_py_path = Path(__file__).parent / "core" / "agent.py"
        _spec = importlib.util.spec_from_file_location("core._agent_module", _agent_py_path)
        _agent_module = importlib.util.module_from_spec(_spec)
        sys.modules["core._agent_module"] = _agent_module
        _spec.loader.exec_module(_agent_module)
        globals()[name] = getattr(_agent_module, name)
        return globals()[name]
    elif name in ("Config", "load_config"):
        from utils.config import Config, load_config
        globals()[name] = Config if name == "Config" else load_config
        return globals()[name]
    elif name in ("KnowledgeMemory", "MemoryItem"):
        from utils.memory import KnowledgeMemory, MemoryItem
        globals()[name] = KnowledgeMemory if name == "KnowledgeMemory" else MemoryItem
        return globals()[name]
    elif name == "APIClient":
        from utils.api_client import APIClient
        globals()[name] = APIClient
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Simulacrum",
    "AgentState",
    "Config",
    "load_config",
    "KnowledgeMemory",
    "MemoryItem",
    "APIClient",
]