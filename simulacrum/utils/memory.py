"""Simulacrum 知识记忆系统"""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class MemoryItem:
    """记忆条目"""
    id: str
    content: str
    embedding: np.ndarray | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5  # 重要性评分 0-1
    tags: list[str] = field(default_factory=list)
    source: str = "internal"


@dataclass
class Experience:
    """经验条目 (用于好奇心计算)"""
    state: np.ndarray
    action: str
    reward: float
    next_state: np.ndarray
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class KnowledgeMemory:
    """知识记忆库

    存储:
    - 长期记忆: 结构化知识
    - 经验缓冲: 探索历史
    """

    def __init__(self, max_size: int = 1000, memory_path: str = "memory.json"):
        self.max_size = max_size
        self.memory_path = memory_path
        self.memories: list[MemoryItem] = []
        self.experiences: list[Experience] = []
        self.stats = {
            "total_memories": 0,
            "total_experiences": 0,
            "unique_tags": set()
        }

        self._load()

    def add_memory(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        source: str = "internal"
    ) -> str:
        """添加新记忆"""
        import uuid
        mem_id = str(uuid.uuid4())[:8]

        memory = MemoryItem(
            id=mem_id,
            content=content,
            importance=importance,
            tags=tags or [],
            source=source
        )

        self.memories.append(memory)
        self.stats["total_memories"] += 1

        for tag in memory.tags:
            self.stats["unique_tags"].add(tag)

        # 保持大小限制
        if len(self.memories) > self.max_size:
            # 删除重要性最低的
            self.memories.sort(key=lambda m: m.importance)
            self.memories.pop(0)

        self._save()
        return mem_id

    def add_experience(
        self,
        state: np.ndarray,
        action: str,
        reward: float,
        next_state: np.ndarray
    ) -> None:
        """添加经验"""
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state
        )
        self.experiences.append(exp)
        self.stats["total_experiences"] += 1

        # 保持大小限制
        if len(self.experiences) > self.max_size:
            self.experiences.pop(0)

    def get_recent_memories(self, n: int = 10) -> list[MemoryItem]:
        """获取最近N条记忆"""
        return self.memories[-n:]

    def search_by_tag(self, tag: str) -> list[MemoryItem]:
        """按标签搜索"""
        return [m for m in self.memories if tag in m.tags]

    def get_importance_distribution(self) -> dict[str, float]:
        """获取重要性分布"""
        if not self.memories:
            return {"low": 0, "medium": 0, "high": 0}

        importances = [m.importance for m in self.memories]
        return {
            "low": sum(1 for i in importances if i < 0.33) / len(importances),
            "medium": sum(1 for i in importances if 0.33 <= i < 0.66) / len(importances),
            "high": sum(1 for i in importances if i >= 0.66) / len(importances)
        }

    def state_dict(self) -> dict[str, Any]:
        """获取状态字典 (用于序列化)"""
        return {
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "importance": m.importance,
                    "tags": m.tags,
                    "source": m.source
                }
                for m in self.memories
            ],
            "stats": {
                "total_memories": self.stats["total_memories"],
                "total_experiences": self.stats["total_experiences"],
                "unique_tags": list(self.stats["unique_tags"])
            }
        }

    def _save(self) -> None:
        """保存到文件"""
        try:
            with open(self.memory_path, 'w', encoding='utf-8') as f:
                json.dump(self.state_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] memory save failed: {e}")

    def _load(self) -> None:
        """从文件加载"""
        if not os.path.exists(self.memory_path):
            return

        try:
            with open(self.memory_path, encoding='utf-8') as f:
                data = json.load(f)

            self.memories = [
                MemoryItem(
                    id=m["id"],
                    content=m["content"],
                    timestamp=m.get("timestamp", ""),
                    importance=m.get("importance", 0.5),
                    tags=m.get("tags", []),
                    source=m.get("source", "internal")
                )
                for m in data.get("memories", [])
            ]

            if "unique_tags" in data.get("stats", {}):
                self.stats["unique_tags"] = set(data["stats"]["unique_tags"])

        except Exception as e:
            print(f"[WARN] memory load failed: {e}")

    def clear(self) -> None:
        """清空记忆"""
        self.memories.clear()
        self.experiences.clear()
        self.stats = {"total_memories": 0, "total_experiences": 0, "unique_tags": set()}
        self._save()
