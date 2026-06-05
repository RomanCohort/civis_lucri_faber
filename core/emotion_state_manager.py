# =============================================================================
# Emotion State Manager - 情绪状态管理器
# =============================================================================
# 情绪状态的持久化和恢复
#
# 功能：
# 1. EmotionStateManager - 状态管理器
# 2. 状态保存/加载 (JSON序列化)
# 3. 会话级别状态管理
# 4. 情绪历史记录
# 5. 状态快照功能
#
# 事件驱动:
# - 订阅 EMOTION_UPDATED: 自动记录状态变化
# - 订阅 HIBERNATE_ENTER: 进入休眠时自动保存状态
# - 订阅 HIBERNATE_EXIT: 退出休眠时自动加载状态
# =============================================================================

import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

from core.events import EMOTION_UPDATED, HIBERNATE_ENTER, HIBERNATE_EXIT

logger = logging.getLogger(__name__)


# =============================================================================
# 快照数据结构
# =============================================================================

@dataclass
class EmotionSnapshot:
    """情绪状态快照"""
    snapshot_id: str
    timestamp: str
    label: str = ""
    state: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class EmotionHistoryEntry:
    """情绪历史记录条目"""
    timestamp: str
    emotion: str
    intensity: float
    mood_valence: float
    mood_arousal: float
    mood_dominance: float
    social_emotion: str
    regulation_capacity: float
    criticality: float


# =============================================================================
# 情绪状态管理器
# =============================================================================

class EmotionStateManager:
    """
    情绪状态管理器

    功能:
    1. save_state() - 保存当前状态到 JSON 文件
    2. load_state() - 从 JSON 文件加载状态
    3. get_history() - 获取历史记录
    4. take_snapshot() - 创建快照 (带标签)
    5. restore_snapshot() - 恢复快照
    6. clear_history() - 清空历史
    7. export_history() - 导出历史到文件

    与 IntegratedAdvancedEmotionSystem 集成:
    - 在每次 emotion_process 后自动记录历史
    - 支持休眠唤醒时的状态持久化

    JSON 序列化:
    - AdvancedEmotionState 自动转换为 dict
    - timestamp 使用 ISO 格式
    - 支持增量保存 (仅保存变化)
    """

    def __init__(
        self,
        emotion_system=None,
        save_dir: str | Path = "./emotion_states",
        history_maxlen: int = 1000,
        snapshot_maxlen: int = 50,
        auto_save: bool = False,
        event_bus=None,
    ):
        """
        初始化情绪状态管理器

        Args:
            emotion_system: IntegratedAdvancedEmotionSystem 实例 (可选)
            save_dir: 状态保存目录
            history_maxlen: 历史记录最大长度
            snapshot_maxlen: 快照最大数量
            auto_save: 是否自动保存 (每次更新后)
            event_bus: EventBus 实例 (用于事件订阅)
        """
        self.emotion_system = emotion_system
        self.save_dir = Path(save_dir)
        self.history_maxlen = history_maxlen
        self.snapshot_maxlen = snapshot_maxlen
        self.auto_save = auto_save
        self._bus = event_bus

        # 历史记录
        self.history: deque[EmotionHistoryEntry] = deque(maxlen=history_maxlen)

        # 快照列表
        self.snapshots: deque[EmotionSnapshot] = deque(maxlen=snapshot_maxlen)

        # 当前状态缓存
        self._current_state: dict = {}

        # 会话 ID (用于区分不同会话)
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 上次保存时间
        self._last_save_time: float = 0.0

        # 状态文件路径
        self._state_file = self.save_dir / f"emotion_state_{self.session_id}.json"
        self._history_file = self.save_dir / f"emotion_history_{self.session_id}.json"
        self._snapshot_dir = self.save_dir / "snapshots"

        # 创建目录
        self._ensure_dirs()

        # 事件订阅
        if self._bus is not None:
            self._bus.subscribe(EMOTION_UPDATED, self.on_emotion_updated, priority=0, name="emotion_state_manager")
            self._bus.subscribe(HIBERNATE_ENTER, self.on_hibernate_enter, priority=0, name="emotion_state_manager")
            self._bus.subscribe(HIBERNATE_EXIT, self.on_hibernate_exit, priority=0, name="emotion_state_manager")

    def _ensure_dirs(self) -> None:
        """确保保存目录存在"""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 状态保存/加载
    # =========================================================================

    def save_state(
        self,
        state: Optional["AdvancedEmotionState"] = None,
        filepath: Optional[str | Path] = None,
        incremental: bool = False,
    ) -> dict[str, Any]:
        """
        保存情绪状态

        Args:
            state: AdvancedEmotionState 实例 (如果为 None, 从 emotion_system 获取)
            filepath: 保存路径 (如果为 None, 使用默认路径)
            incremental: 是否增量保存 (仅保存变化字段)

        Returns:
            result: 保存结果
        """
        # 获取状态
        if state is None:
            if self.emotion_system is not None:
                # 从 emotion_system 获取最新状态
                try:
                    if len(self.emotion_system.state_history) > 0:
                        last_entry = self.emotion_system.state_history[-1]
                        # 构造 AdvancedEmotionState
                        adv_state = self._extract_state_from_history(last_entry)
                        state_dict = self._state_to_dict(adv_state)
                    else:
                        state_dict = self._current_state
                except (AttributeError, KeyError):
                    state_dict = self._current_state
            else:
                state_dict = self._current_state
        else:
            state_dict = self._state_to_dict(state)

        # 增量保存
        if incremental and self._current_state:
            changed_fields = {}
            for key, value in state_dict.items():
                if key not in self._current_state or self._current_state[key] != value:
                    changed_fields[key] = value
            state_dict = changed_fields

        # 添加元数据
        state_dict["_metadata"] = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "save_type": "incremental" if incremental else "full",
        }

        # 确定保存路径
        save_path = Path(filepath) if filepath else self._state_file

        # 保存到文件
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2, ensure_ascii=False)

            self._last_save_time = time.time()
            self._current_state.update(state_dict)

            logger.info(f"Emotion state saved to {save_path}")

            return {
                "success": True,
                "filepath": str(save_path),
                "timestamp": state_dict["_metadata"]["timestamp"],
                "fields_saved": len(state_dict) - 1,  # 减去 _metadata
            }
        except (IOError, json.JSONEncodeError) as e:
            logger.error(f"Failed to save emotion state: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def load_state(
        self,
        filepath: Optional[str | Path] = None,
        apply_to_system: bool = True,
    ) -> dict[str, Any]:
        """
        加载情绪状态

        Args:
            filepath: 加载路径 (如果为 None, 使用默认路径)
            apply_to_system: 是否将状态应用到 emotion_system

        Returns:
            result: 加载结果
        """
        # 确定加载路径
        load_path = Path(filepath) if filepath else self._state_file

        if not load_path.exists():
            logger.warning(f"State file not found: {load_path}")
            return {
                "success": False,
                "error": "file_not_found",
                "filepath": str(load_path),
            }

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                state_dict = json.load(f)

            # 提取元数据
            metadata = state_dict.pop("_metadata", {})

            # 更新当前状态
            self._current_state.update(state_dict)

            # 应用到 emotion_system
            if apply_to_system and self.emotion_system is not None:
                self._apply_state_to_system(state_dict)

            logger.info(f"Emotion state loaded from {load_path}")

            return {
                "success": True,
                "filepath": str(load_path),
                "state": state_dict,
                "metadata": metadata,
            }
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load emotion state: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    # =========================================================================
    # 历史记录
    # =========================================================================

    def record_history(
        self,
        state: Optional["AdvancedEmotionState"] = None,
    ) -> None:
        """
        记录情绪历史

        Args:
            state: AdvancedEmotionState 实例
        """
        if state is None:
            # 使用当前缓存状态
            state_dict = self._current_state
        else:
            state_dict = self._state_to_dict(state)

        entry = EmotionHistoryEntry(
            timestamp=datetime.now().isoformat(),
            emotion=state_dict.get("current_emotion", "neutral"),
            intensity=state_dict.get("emotion_intensity", 0.0),
            mood_valence=state_dict.get("mood_valence", 0.0),
            mood_arousal=state_dict.get("mood_arousal", 0.0),
            mood_dominance=state_dict.get("mood_dominance", 0.0),
            social_emotion=state_dict.get("social_emotion", "neutral"),
            regulation_capacity=state_dict.get("regulation_capacity", 1.0),
            criticality=state_dict.get("criticality", 0.0),
        )

        self.history.append(entry)

        # 自动保存
        if self.auto_save:
            self._auto_save_check()

    def get_history(
        self,
        limit: int = 100,
        since: Optional[str] = None,
        emotion_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        获取历史记录

        Args:
            limit: 返回条目数量上限
            since: 时间起点 (ISO 格式)
            emotion_filter: 情绪类型过滤

        Returns:
            history: 历史记录列表
        """
        result = []

        for entry in reversed(self.history):
            # 时间过滤
            if since and entry.timestamp < since:
                continue

            # 情绪过滤
            if emotion_filter and entry.emotion != emotion_filter:
                continue

            result.append(asdict(entry))

            if len(result) >= limit:
                break

        return result

    def clear_history(self) -> dict[str, Any]:
        """
        清空历史记录

        Returns:
            result: 清空结果
        """
        cleared_count = len(self.history)
        self.history.clear()

        logger.info(f"Emotion history cleared: {cleared_count} entries")

        return {
            "success": True,
            "cleared_count": cleared_count,
        }

    def export_history(
        self,
        filepath: Optional[str | Path] = None,
        format: str = "json",
    ) -> dict[str, Any]:
        """
        导出历史记录到文件

        Args:
            filepath: 导出路径
            format: 导出格式 (json/csv)

        Returns:
            result: 导出结果
        """
        if not self.history:
            return {
                "success": False,
                "error": "no_history",
            }

        export_path = Path(filepath) if filepath else self._history_file

        try:
            if format == "json":
                history_data = [asdict(entry) for entry in self.history]
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(history_data, f, indent=2, ensure_ascii=False)
            elif format == "csv":
                import csv
                with open(export_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "timestamp", "emotion", "intensity",
                        "mood_valence", "mood_arousal", "mood_dominance",
                        "social_emotion", "regulation_capacity", "criticality",
                    ])
                    writer.writeheader()
                    for entry in self.history:
                        writer.writerow(asdict(entry))
            else:
                return {
                    "success": False,
                    "error": f"unsupported_format: {format}",
                }

            logger.info(f"History exported to {export_path}")

            return {
                "success": True,
                "filepath": str(export_path),
                "format": format,
                "entries_exported": len(self.history),
            }
        except IOError as e:
            logger.error(f"Failed to export history: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    # =========================================================================
    # 快照功能
    # =========================================================================

    def take_snapshot(
        self,
        label: str = "",
        state: Optional["AdvancedEmotionState"] = None,
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        创建状态快照

        Args:
            label: 快照标签 (如 "therapy_session_1", "pre_treatment")
            state: AdvancedEmotionState 实例
            metadata: 额外元数据

        Returns:
            result: 快照创建结果
        """
        snapshot_id = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # 获取状态
        if state is None:
            state_dict = self._current_state.copy()
        else:
            state_dict = self._state_to_dict(state)

        snapshot = EmotionSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now().isoformat(),
            label=label,
            state=state_dict,
            metadata=metadata or {},
        )

        self.snapshots.append(snapshot)

        # 保存快照文件
        snapshot_file = self._snapshot_dir / f"{snapshot_id}.json"
        try:
            with open(snapshot_file, "w", encoding="utf-8") as f:
                json.dump(asdict(snapshot), f, indent=2, ensure_ascii=False)

            logger.info(f"Snapshot created: {snapshot_id} (label: {label})")

            return {
                "success": True,
                "snapshot_id": snapshot_id,
                "label": label,
                "filepath": str(snapshot_file),
            }
        except IOError as e:
            logger.error(f"Failed to save snapshot: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def restore_snapshot(
        self,
        snapshot_id: Optional[str] = None,
        label: Optional[str] = None,
        apply_to_system: bool = True,
    ) -> dict[str, Any]:
        """
        恢复快照

        Args:
            snapshot_id: 快照 ID
            label: 快照标签 (如果提供, 查找匹配标签的最新快照)
            apply_to_system: 是否应用到 emotion_system

        Returns:
            result: 恢复结果
        """
        # 查找快照
        snapshot = None

        if snapshot_id:
            # 按 ID 查找
            for snap in self.snapshots:
                if snap.snapshot_id == snapshot_id:
                    snapshot = snap
                    break

            # 如果内存中没找到, 从文件加载
            if snapshot is None:
                snapshot_file = self._snapshot_dir / f"{snapshot_id}.json"
                if snapshot_file.exists():
                    try:
                        with open(snapshot_file, "r", encoding="utf-8") as f:
                            snapshot_data = json.load(f)
                        snapshot = EmotionSnapshot(**snapshot_data)
                    except (IOError, json.JSONDecodeError):
                        pass
        elif label:
            # 按标签查找最新
            for snap in reversed(self.snapshots):
                if snap.label == label:
                    snapshot = snap
                    break

        if snapshot is None:
            return {
                "success": False,
                "error": "snapshot_not_found",
                "snapshot_id": snapshot_id,
                "label": label,
            }

        # 恢复状态
        self._current_state.update(snapshot.state)

        # 应用到 emotion_system
        if apply_to_system and self.emotion_system is not None:
            self._apply_state_to_system(snapshot.state)

        logger.info(f"Snapshot restored: {snapshot.snapshot_id}")

        return {
            "success": True,
            "snapshot_id": snapshot.snapshot_id,
            "label": snapshot.label,
            "timestamp": snapshot.timestamp,
            "state": snapshot.state,
        }

    def list_snapshots(
        self,
        label_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        列出所有快照

        Args:
            label_filter: 标签过滤

        Returns:
            snapshots: 快照列表
        """
        result = []
        for snap in self.snapshots:
            if label_filter and snap.label != label_filter:
                continue
            result.append({
                "snapshot_id": snap.snapshot_id,
                "timestamp": snap.timestamp,
                "label": snap.label,
                "has_state": bool(snap.state),
            })
        return result

    def delete_snapshot(
        self,
        snapshot_id: str,
    ) -> dict[str, Any]:
        """
        删除快照

        Args:
            snapshot_id: 快照 ID

        Returns:
            result: 删除结果
        """
        # 从内存删除
        for i, snap in enumerate(self.snapshots):
            if snap.snapshot_id == snapshot_id:
                self.snapshots.remove(snap)
                break

        # 从文件删除
        snapshot_file = self._snapshot_dir / f"{snapshot_id}.json"
        if snapshot_file.exists():
            try:
                snapshot_file.unlink()
                logger.info(f"Snapshot deleted: {snapshot_id}")
                return {
                    "success": True,
                    "snapshot_id": snapshot_id,
                }
            except IOError as e:
                return {
                    "success": False,
                    "error": str(e),
                }

        return {
            "success": True,
            "snapshot_id": snapshot_id,
            "note": "file_not_found",
        }

    # =========================================================================
    # 事件处理
    # =========================================================================

    def on_emotion_updated(self, event) -> dict[str, Any]:
        """
        事件驱动: 响应 EMOTION_UPDATED，自动记录状态变化

        Args:
            event: EMOTION_UPDATED 事件

        Returns:
            result: 记录结果
        """
        # 从事件数据提取情绪状态
        emotion_state = event.data.get("emotion_state", {})

        if emotion_state:
            # 更新当前状态缓存
            self._current_state.update(emotion_state)

            # 记录到历史
            entry = EmotionHistoryEntry(
                timestamp=datetime.now().isoformat(),
                emotion=emotion_state.get("current_emotion", "neutral"),
                intensity=emotion_state.get("emotion_intensity", 0.0),
                mood_valence=emotion_state.get("mood_valence", 0.0),
                mood_arousal=emotion_state.get("mood_arousal", 0.0),
                mood_dominance=emotion_state.get("mood_dominance", 0.0),
                social_emotion=emotion_state.get("social_emotion", "neutral"),
                regulation_capacity=emotion_state.get("regulation_capacity", 1.0),
                criticality=emotion_state.get("criticality", 0.0),
            )
            self.history.append(entry)

            logger.debug(f"Emotion state updated: {emotion_state.get('current_emotion')}")

            # 自动保存检查
            if self.auto_save:
                self._auto_save_check()

        return {
            "recorded": bool(emotion_state),
            "emotion": emotion_state.get("current_emotion", "none"),
        }

    def on_hibernate_enter(self, event) -> dict[str, Any]:
        """
        事件驱动: 进入休眠时保存状态

        Args:
            event: HIBERNATE_ENTER 事件

        Returns:
            result: 保存结果
        """
        logger.info("Saving emotion state before hibernation")

        # 保存当前状态
        save_result = self.save_state()

        # 导出历史
        export_result = self.export_history()

        return {
            "state_saved": save_result.get("success", False),
            "history_exported": export_result.get("success", False),
        }

    def on_hibernate_exit(self, event) -> dict[str, Any]:
        """
        事件驱动: 退出休眠时加载状态

        Args:
            event: HIBERNATE_EXIT 事件

        Returns:
            result: 加载结果
        """
        logger.info("Loading emotion state after hibernation")

        # 加载状态
        load_result = self.load_state()

        return {
            "state_loaded": load_result.get("success", False),
        }

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _state_to_dict(self, state: "AdvancedEmotionState") -> dict:
        """
        将 AdvancedEmotionState 转换为 dict

        Args:
            state: AdvancedEmotionState 实例

        Returns:
            state_dict: 状态字典
        """
        # AdvancedEmotionState 是 dataclass, 可以用 asdict
        from core.advanced_emotion_integration import AdvancedEmotionState

        if isinstance(state, AdvancedEmotionState):
            return asdict(state)

        # 兜底处理
        if hasattr(state, "__dict__"):
            return {
                "current_emotion": getattr(state, "current_emotion", "neutral"),
                "emotion_intensity": getattr(state, "emotion_intensity", 0.0),
                "mood_valence": getattr(state, "mood_valence", 0.0),
                "mood_arousal": getattr(state, "mood_arousal", 0.0),
                "mood_dominance": getattr(state, "mood_dominance", 0.0),
                "social_emotion": getattr(state, "social_emotion", "neutral"),
                "social_intensity": getattr(state, "social_intensity", 0.0),
                "regulation_capacity": getattr(state, "regulation_capacity", 1.0),
                "regulation_strategy": getattr(state, "regulation_strategy", "none"),
                "emotion_velocity": getattr(state, "emotion_velocity", 0.0),
                "criticality": getattr(state, "criticality", 0.0),
            }

        return {}

    def _dict_to_state(self, state_dict: dict) -> "AdvancedEmotionState":
        """
        将 dict 转换为 AdvancedEmotionState

        Args:
            state_dict: 状态字典

        Returns:
            state: AdvancedEmotionState 实例
        """
        from core.advanced_emotion_integration import AdvancedEmotionState

        return AdvancedEmotionState(
            current_emotion=state_dict.get("current_emotion", "neutral"),
            emotion_intensity=state_dict.get("emotion_intensity", 0.0),
            mood_valence=state_dict.get("mood_valence", 0.0),
            mood_arousal=state_dict.get("mood_arousal", 0.0),
            mood_dominance=state_dict.get("mood_dominance", 0.0),
            social_emotion=state_dict.get("social_emotion", "neutral"),
            social_intensity=state_dict.get("social_intensity", 0.0),
            regulation_capacity=state_dict.get("regulation_capacity", 1.0),
            regulation_strategy=state_dict.get("regulation_strategy", "none"),
            emotion_velocity=state_dict.get("emotion_velocity", 0.0),
            criticality=state_dict.get("criticality", 0.0),
        )

    def _extract_state_from_history(self, history_entry: dict) -> "AdvancedEmotionState":
        """
        从历史记录提取状态

        Args:
            history_entry: emotion_system.state_history 中的条目

        Returns:
            state: AdvancedEmotionState 实例
        """
        from core.advanced_emotion_integration import AdvancedEmotionState

        # history_entry 可能包含 emotion, mood, social 等字段
        emotion = history_entry.get("emotion", None)
        mood = history_entry.get("mood", None)
        social = history_entry.get("social", "neutral")

        # 提取情绪强度
        intensity = 0.0
        if emotion is not None and isinstance(emotion, torch.Tensor):
            intensity = emotion.max().item() if emotion.numel() > 0 else 0.0

        # 提取心境
        mood_valence = 0.0
        mood_arousal = 0.0
        mood_dominance = 0.0
        if mood is not None:
            mood_valence = getattr(mood, "valence", 0.0)
            mood_arousal = getattr(mood, "arousal", 0.0)
            mood_dominance = getattr(mood, "dominance", 0.0)

        return AdvancedEmotionState(
            current_emotion=self._get_emotion_name(emotion),
            emotion_intensity=intensity,
            mood_valence=mood_valence,
            mood_arousal=mood_arousal,
            mood_dominance=mood_dominance,
            social_emotion=social if isinstance(social, str) else "neutral",
        )

    def _get_emotion_name(self, emotion: Optional[torch.Tensor]) -> str:
        """
        从情绪张量获取情绪名称

        Args:
            emotion: 情绪张量 [B, emotion_dim]

        Returns:
            emotion_name: 情绪名称
        """
        if emotion is None or not isinstance(emotion, torch.Tensor):
            return "neutral"

        emotions = [
            'joy', 'sadness', 'anger', 'fear',
            'surprise', 'disgust', 'neutral', 'anticipation'
        ]

        idx = emotion.argmax().item()
        return emotions[idx] if idx < len(emotions) else 'neutral'

    def _apply_state_to_system(self, state_dict: dict) -> None:
        """
        将状态应用到 emotion_system

        Args:
            state_dict: 状态字典
        """
        if self.emotion_system is None:
            return

        # 应用心境状态到 mood_system
        if hasattr(self.emotion_system, "mood_system"):
            mood_system = self.emotion_system.mood_system
            if mood_system is not None and hasattr(mood_system, "mood_buffer"):
                # 构造心境张量
                mood_tensor = torch.tensor([
                    state_dict.get("mood_valence", 0.0),
                    state_dict.get("mood_arousal", 0.0),
                    state_dict.get("mood_dominance", 0.0),
                    0.0,  # activation
                    0.0,  # pleasantness
                ]).unsqueeze(0)
                mood_system.mood_buffer.append(mood_tensor)

        logger.info("Applied loaded state to emotion_system")

    def _auto_save_check(self) -> None:
        """
        自动保存检查
        """
        # 每 60 秒最多自动保存一次
        if time.time() - self._last_save_time > 60:
            self.save_state(incremental=True)

    # =========================================================================
    # 摘要和统计
    # =========================================================================

    def get_summary(self) -> dict:
        """
        获取状态管理器摘要

        Returns:
            summary: 摘要信息
        """
        # 统计历史
        emotion_counts = {}
        for entry in self.history:
            emotion_counts[entry.emotion] = emotion_counts.get(entry.emotion, 0) + 1

        # 平均值
        avg_valence = 0.0
        avg_arousal = 0.0
        avg_intensity = 0.0
        if self.history:
            avg_valence = sum(e.mood_valence for e in self.history) / len(self.history)
            avg_arousal = sum(e.mood_arousal for e in self.history) / len(self.history)
            avg_intensity = sum(e.intensity for e in self.history) / len(self.history)

        return {
            "session_id": self.session_id,
            "history_count": len(self.history),
            "snapshot_count": len(self.snapshots),
            "emotion_distribution": emotion_counts,
            "avg_valence": avg_valence,
            "avg_arousal": avg_arousal,
            "avg_intensity": avg_intensity,
            "current_emotion": self._current_state.get("current_emotion", "neutral"),
            "save_dir": str(self.save_dir),
            "auto_save": self.auto_save,
            "last_save_time": datetime.fromtimestamp(self._last_save_time).isoformat() if self._last_save_time > 0 else None,
        }


# =============================================================================
# 便捷函数
# =============================================================================

def create_emotion_state_manager(
    emotion_system=None,
    save_dir: str | Path = "./emotion_states",
    event_bus=None,
) -> EmotionStateManager:
    """
    创建情绪状态管理器

    Args:
        emotion_system: IntegratedAdvancedEmotionSystem 实例
        save_dir: 保存目录
        event_bus: EventBus 实例

    Returns:
        manager: EmotionStateManager 实例
    """
    return EmotionStateManager(
        emotion_system=emotion_system,
        save_dir=save_dir,
        event_bus=event_bus,
    )


__all__ = [
    'EmotionStateManager',
    'EmotionSnapshot',
    'EmotionHistoryEntry',
    'create_emotion_state_manager',
]


# =============================================================================
# 测试
# =============================================================================

def test_emotion_state_manager():
    """测试情绪状态管理器"""
    print("=" * 60)
    print("Testing Emotion State Manager")
    print("=" * 60)

    # 创建管理器
    manager = EmotionStateManager(save_dir="./test_emotion_states")

    print("\n[1] Testing state save/load...")
    from core.advanced_emotion_integration import AdvancedEmotionState

    test_state = AdvancedEmotionState(
        current_emotion="joy",
        emotion_intensity=0.7,
        mood_valence=0.5,
        mood_arousal=0.6,
        mood_dominance=0.4,
        social_emotion="admiration",
        regulation_capacity=0.8,
    )

    save_result = manager.save_state(test_state)
    print(f"  Save result: {save_result['success']}")

    load_result = manager.load_state()
    print(f"  Load result: {load_result['success']}")
    print(f"  Loaded emotion: {load_result['state'].get('current_emotion')}")

    print("\n[2] Testing history...")
    manager.record_history(test_state)
    manager.record_history(AdvancedEmotionState(current_emotion="sadness", mood_valence=-0.3))
    history = manager.get_history(limit=5)
    print(f"  History entries: {len(history)}")
    for entry in history[:2]:
        print(f"    - {entry['emotion']} @ {entry['timestamp']}")

    print("\n[3] Testing snapshots...")
    snap1 = manager.take_snapshot(label="pre_test", state=test_state)
    print(f"  Snapshot created: {snap1['snapshot_id']}")

    snap_list = manager.list_snapshots()
    print(f"  Snapshots: {len(snap_list)}")

    restore_result = manager.restore_snapshot(label="pre_test")
    print(f"  Restore result: {restore_result['success']}")

    print("\n[4] Testing export...")
    export_result = manager.export_history(format="json")
    print(f"  Export result: {export_result['success']}")

    print("\n[5] Summary...")
    summary = manager.get_summary()
    print(f"  History count: {summary['history_count']}")
    print(f"  Emotion distribution: {summary['emotion_distribution']}")

    # 清理测试文件
    import shutil
    if os.path.exists("./test_emotion_states"):
        shutil.rmtree("./test_emotion_states")

    print("\n" + "=" * 60)
    print("Emotion state manager working!")
    print("=" * 60)


if __name__ == "__main__":
    test_emotion_state_manager()