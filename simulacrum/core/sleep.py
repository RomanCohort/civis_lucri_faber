"""
睡眠系统 (Sleep System)

对应生物睡眠的核心功能：
1. 睡眠周期 - REM/NREM交替
2. 记忆巩固 - 优先Replay
3. 突触缩减 - 能量恢复
4. 皮层回放 - 离线学习

这是最具类脑特征的机制，也是最易展示效果的。
"""
import random
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SleepStage(Enum):
    """睡眠阶段"""
    AWAKE = "awake"
    NREM1 = "nrem1"      # 浅睡眠
    NREM2 = "nrem2"      # 中睡眠
    NREM3 = "deep"         # 深度睡眠
    REM = "rem"            # 快速眼动（做梦）
    HYPNAGOGIC = "hypnagogic"  # 入睡期


@dataclass
class SleepCycle:
    """完整睡眠周期"""
    stages: list[SleepStage] = field(default_factory=list)
    current_stage: SleepStage = SleepStage.AWAKE
    cycle_count: int = 0

    # 时间统计
    total_sleep_time: float = 0.0
    rem_ratio: float = 0.25      # REM约占25%
    nrem2_ratio: float = 0.45   # NREM2约占45%
    nrem3_ratio: float = 0.20    # NREM3约占20%
    nrem1_ratio: float = 0.10   # NREM1约占10%


@dataclass
class MemoryReplay:
    """记忆回放单元"""
    state: np.ndarray
    action: str
    reward: float
    next_state: np.ndarray
    priority: float  # replay优先级
    timestamp: int


class SleepController(nn.Module):
    """
    睡眠控制器

    管理睡眠周期转换，模拟生物睡眠节律：
    - AWAKE → NREM1 → NREM2 → NREM3 → NREM2 → REM → NREM2 → ...
    """

    def __init__(
        self,
        # 周期参数
        awake_to_sleep_threshold: float = 0.8,  # 转入睡眠的疲惫度阈值
        sleep_cycle_duration: int = 90,           # 完整周期(分钟) - 人类约90min
        rem_duration: int = 20,                 # REM时长(分钟)
        nrem2_duration: int = 40,              # NREM2时长
        nrem3_duration: int = 30,              # 深度睡眠时长

        # 控制参数
        enable_dream: bool = True,              # 是否启用梦境生成
        dream_creativity: float = 0.3,          # 梦境创造力
    ):
        super().__init__()

        # 周期参数
        self.awake_threshold = awake_to_sleep_threshold
        self.cycle_duration = sleep_cycle_duration
        self.rem_duration = rem_duration
        self.nrem2_duration = nrem2_duration
        self.nrem3_duration = nrem3_duration

        # 梦境
        self.enable_dream = enable_dream
        self.dream_creativity = dream_creativity

        # 状态
        self.fatigue = 0.0           # 疲惫度 (0-1)
        self.is_sleeping = False
        self.current_cycle = SleepCycle()
        self.cycle_progress = 0.0     # 当前周期进度(0-1)
        self.total_cycles = 0

        # 时间追踪
        self.step_count = 0
        self.last_sleep_stage = SleepStage.AWAKE

        # 梦境历史
        self.dream_history: list[dict] = []

        # 统计
        self.sleep_stats = {
            'total_sleep_time': 0,
            'rem_count': 0,
            'nrem3_count': 0,
            'dream_count': 0,
        }

    def update_fatigue(self, info_gain_reward: float, step_duration: float = 1.0):
        """
        更新疲惫度

        信息增益消耗能量增加疲惫
        空闲时间长 → 高疲惫
        """
        self.step_count += 1

        # 信息增益消耗energy（思考会累）
        info_gain_effect = info_gain_reward * 0.1

        # 空闲增加疲惫
        idle_effect = step_duration * 0.05

        self.fatigue = np.clip(self.fatigue + idle_effect + info_gain_effect, 0, 1)

    def should_sleep(self) -> bool:
        """判断是否应该入睡"""
        return self.fatigue > self.awake_threshold

    def _get_next_stage(self, current: SleepStage) -> SleepStage:
        """获取下一阶段"""
        if current == SleepStage.AWAKE:
            return SleepStage.NREM1
        elif current == SleepStage.NREM1:
            return SleepStage.NREM2
        elif current == SleepStage.NREM2:
            # 深度睡眠倾向随周期递减 (模拟生物: 后半夜深睡减少)
            deep_tendency = max(0.3, 0.9 - 0.12 * self.total_cycles)
            # 确定性选择: cycle_progress < deep_tendency → NREM3
            if self.cycle_progress < deep_tendency:
                return SleepStage.NREM3
            else:
                return SleepStage.REM
        elif current == SleepStage.NREM3 or current == SleepStage.REM:
            return SleepStage.NREM2
        elif current == SleepStage.HYPNAGOGIC:
            return SleepStage.NREM1
        return SleepStage.NREM1

    def step(self) -> SleepStage:
        """
        执行睡眠周期步骤

        Returns:
            current_stage: 当前睡眠阶段
        """
        if not self.is_sleeping:
            # 保持清醒
            if self.should_sleep():
                self.enter_sleep()
            return SleepStage.AWAKE

        # 睡眠中：推进周期
        self.cycle_progress += 1 / self.cycle_duration

        # 检查是否完成一个周期
        if self.cycle_progress >= 1.0:
            self.cycle_progress = 0.0
            self.total_cycles += 1
            self.current_cycle.cycle_count = self.total_cycles

        # 阶段转换
        if self.cycle_progress >= self._get_stage_threshold(self.current_cycle.current_stage):
            self.last_sleep_stage = self.current_cycle.current_stage
            self.current_cycle.current_stage = self._get_next_stage(self.current_cycle.current_stage)

            # 统计
            if self.current_cycle.current_stage == SleepStage.REM:
                self.sleep_stats['rem_count'] += 1
            elif self.current_cycle.current_stage == SleepStage.NREM3:
                self.sleep_stats['nrem3_count'] += 1

        # 更新统计
        self.sleep_stats['total_sleep_time'] += 1

        return self.current_cycle.current_stage

    def _get_stage_threshold(self, stage: SleepStage) -> float:
        """获取各阶段时间阈值 (基于duration的比例)"""
        total = self.nrem2_duration + self.nrem3_duration + self.rem_duration

        if stage == SleepStage.NREM1:
            return 0.1
        elif stage == SleepStage.NREM2:
            return (self.nrem2_duration / total) * 0.8
        elif stage == SleepStage.NREM3:
            return ((self.nrem2_duration + self.nrem3_duration) / total) * 0.8
        elif stage == SleepStage.REM:
            return 0.95
        return 0.0

    def enter_sleep(self):
        """进入睡眠状态"""
        self.is_sleeping = True
        self.current_cycle = SleepCycle()
        self.current_cycle.current_stage = SleepStage.NREM1
        self.cycle_progress = 0.0

    def wake_up(self):
        """清醒"""
        self.is_sleeping = False
        self.fatigue = 0.1  # 睡醒后精力恢复
        self.last_sleep_stage = self.current_cycle.current_stage
        self.current_cycle.current_stage = SleepStage.AWAKE

    def get_consolidation_bonus(self) -> float:
        """
        获取记忆巩固增益 (连续调制)

        不同阶段有不同效果，阶段内随进度连续变化
        """
        if not self.is_sleeping:
            return 0.0

        current = self.current_cycle.current_stage
        # 阶段内连续调制: 中期增益最高 (正弦包络)
        modulation = 0.1 * np.sin(np.pi * self.cycle_progress)

        if current == SleepStage.REM:
            return 0.7 + 0.2 * modulation  # 情绪/创造性
        elif current == SleepStage.NREM2:
            return 0.4 + 0.2 * modulation  # 程序性技能
        elif current == SleepStage.NREM3:
            return 0.8 + 0.15 * modulation  # 陈述性记忆最强
        elif current == SleepStage.NREM1:
            return 0.15 + 0.1 * modulation

        return 0.0

    def generate_dream(self, memory: list[MemoryReplay]) -> str | None:
        """
        生成梦境（梦境重构）

        从记忆中随机采样并重组，模拟海阔天空的思维方式
        """
        if not self.enable_dream or not memory:
            return None

        # 选择若干记忆碎片
        n_fragments = min(3, len(memory))
        samples = random.sample(memory, n_fragments)

        # 重组
        dream_elements = []
        for m in samples:
            # 随机打乱细节，加上"创造力"
            if random.random() < self.dream_creativity:
                dream_elements.append(f"[重组] {m.action}")
            else:
                dream_elements.append(m.action)

        dream = " + ".join(dream_elements)

        self.dream_history.append({
            'dream': dream,
            'stage': self.current_cycle.current_stage.value,
            'step': self.step_count,
        })
        self.sleep_stats['dream_count'] += 1

        return dream

    def get_synaptic_downscale_factor(self) -> float:
        """
        获取突触缩减因子 (连续调制)

        深度睡眠时缩减最强，阶段内连续变化
        """
        if not self.is_sleeping:
            return 1.0

        current = self.current_cycle.current_stage
        progress_mod = 0.02 * np.sin(np.pi * self.cycle_progress)

        if current == SleepStage.NREM3:
            return 0.88 + progress_mod  # 强缩减
        elif current == SleepStage.NREM2:
            return 0.93 + progress_mod  # 中缩减
        elif current == SleepStage.REM:
            return 0.97 + progress_mod  # REM期间保持
        else:
            return 0.99 + progress_mod

    def get_summary(self) -> dict:
        """获取摘要"""
        return {
            'is_sleeping': self.is_sleeping,
            'current_stage': self.current_cycle.current_stage.value,
            'fatigue': self.fatigue,
            'cycle_progress': self.cycle_progress,
            'total_cycles': self.total_cycles,
            'consolidation_bonus': self.get_consolidation_bonus(),
            'synaptic_downscale': self.get_synaptic_downscale_factor(),
            'stats': self.sleep_stats,
        }


class MemoryReplayer:
    """
    记忆回放系统

    模拟海马体的记忆巩固机制：
    1. 优先回放高优先级经验
    2. 在睡眠期间进行离线学习
    3. 调整经验优先级
    """

    def __init__(
        self,
        priority_alpha: float = 0.6,   # 优先级指数
        replay_rate: float = 0.1,      # 回放率
        batch_size: int = 32,
    ):
        self.alpha = priority_alpha
        self.replay_rate = replay_rate
        self.batch_size = batch_size

        self.replay_buffer: list[MemoryReplay] = []
        self.replay_count = 0

    def add_experience(
        self,
        state: np.ndarray,
        action: str,
        reward: float,
        next_state: np.ndarray,
    ):
        """添加经验到回放缓冲区"""
        # 计算初始优先级 (基于TD误差)
        priority = abs(reward) + 0.1

        memory = MemoryReplay(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            priority=priority,
            timestamp=0,
        )
        self.replay_buffer.append(memory)

        # 限制缓冲区大小
        if len(self.replay_buffer) > 10000:
            # 删除最低优先级
            self.replay_buffer.sort(key=lambda x: x.priority)
            self.replay_buffer.pop(0)

    def sample(self, n: int = None) -> list[MemoryReplay]:
        """按优先级采样"""
        if not self.replay_buffer:
            return []

        n = n or self.batch_size

        # 按优先级排序后采样
        self.replay_buffer.sort(key=lambda x: x.priority, reverse=True)

        # 概率采样（越靠前概率越大）
        probs = np.array([m.priority for m in self.replay_buffer])
        probs = probs ** self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(
            len(self.replay_buffer),
            size=min(n, len(self.replay_buffer)),
            replace=False,
            p=probs,
        )

        return [self.replay_buffer[i] for i in indices]

    def replay_and_learn(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """回放学习（离线）"""
        if not self.replay_buffer:
            return {'loss': 0.0}

        batch = self.sample()
        if not batch:
            return {'loss': 0.0}

        total_loss = 0.0
        for memory in batch:
            # 模拟学习
            state_tensor = torch.tensor(memory.state, dtype=torch.float32)
            # 简化：预测reward
            pred = model(state_tensor) if hasattr(model, '__call__') else None

            if pred is not None:
                loss = F.mse_loss(pred, torch.tensor([memory.reward]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.replay_count += 1

        return {
            'loss': total_loss / len(batch) if batch else 0.0,
            'replayed': len(batch),
        }

    def update_priorities(self, td_errors: list[float]):
        """更新优先级（基于新TD误差）"""
        for i, error in enumerate(td_errors):
            if i < len(self.replay_buffer):
                # 指数移动平均更新
                self.replay_buffer[i].priority = (
                    self.alpha * self.replay_buffer[i].priority +
                    (1 - self.alpha) * abs(error)
                )

    def get_statistics(self) -> dict:
        """获取统计"""
        return {
            'buffer_size': len(self.replay_buffer),
            'replay_count': self.replay_count,
            'avg_priority': np.mean([m.priority for m in self.replay_buffer]) if self.replay_buffer else 0,
        }


class SleepSystem:
    """
    完整睡眠系统

    整合：
    1. SleepController - 睡眠周期管理
    2. MemoryReplayer - 记忆回放巩固
    3. 皮层Offline学习
    """

    def __init__(
        self,
        enable_sleep: bool = True,
        enable_dream: bool = True,
        sleep_threshold: float = 0.8,
    ):
        self.enable_sleep = enable_sleep

        # 睡眠控制器
        self.controller = SleepController(
            awake_to_sleep_threshold=sleep_threshold,
            enable_dream=enable_dream,
        )

        # 记忆回放
        self.replayer = MemoryReplayer()

        # 离线学习模型（简化）
        self.offline_model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.optimizer = torch.optim.Adam(self.offline_model.parameters(), lr=0.001)

        # 状态
        self.last_stage = SleepStage.AWAKE
        self.dream_content = None

    def update(
        self,
        info_gain_reward: float = 0.0,
        step_duration: float = 1.0,
    ) -> dict:
        """
        更新睡眠系统

        Args:
            info_gain_reward: 信息增益奖励
            step_duration: 步骤时长

        Returns:
            status: 当前状态
        """
        if not self.enable_sleep:
            return {'stage': 'awake', 'is_sleeping': False}

        # 1. 更新疲惫度
        self.controller.update_fatigue(info_gain_reward, step_duration)

        # 2. 执行睡眠周期步骤
        current_stage = self.controller.step()

        # 3. 检查阶段变化
        stage_changed = current_stage != self.last_stage
        self.last_stage = current_stage

        # 4. 不同阶段执行不同操作
        result = {
            'stage': current_stage.value,
            'is_sleeping': self.controller.is_sleeping,
            'stage_changed': stage_changed,
        }

        if current_stage == SleepStage.REM and self.controller.enable_dream:
            # REM：生成梦境
            memories = self.replayer.replay_buffer[-10:]
            self.dream_content = self.controller.generate_dream(memories)
            result['dream'] = self.dream_content

        elif current_stage == SleepStage.NREM3:
            # 深度睡眠：最强制 consolidation
            replay_result = self.replayer.replay_and_learn(
                self.offline_model,
                self.optimizer
            )
            result['offline_loss'] = replay_result.get('loss', 0)

        elif current_stage == SleepStage.AWAKE and self.controller.last_sleep_stage != SleepStage.AWAKE:
            # 刚醒来：恢复精力
            result['fatigue_reset'] = True

        # 5. 获取增益
        result['consolidation_bonus'] = self.controller.get_consolidation_bonus()
        result['synaptic_downscale'] = self.controller.get_synaptic_downscale_factor()

        # 6. 自动清醒条件
        if self.controller.fatigue < 0.2:
            self.controller.wake_up()

        return result

    def add_experience(
        self,
        state: np.ndarray,
        action: str,
        reward: float,
        next_state: np.ndarray,
    ):
        """添加经验用于回放"""
        self.replayer.add_experience(state, action, reward, next_state)

    def get_summary(self) -> dict:
        """获取摘要"""
        return {
            'controller': self.controller.get_summary(),
            'replayer': self.replayer.get_statistics(),
            'dream_content': self.dream_content,
        }


# ============ 便捷函数 ============

def create_sleep_system(
    enable_sleep: bool = True,
    enable_dream: bool = True,
    sleep_threshold: float = 0.8,
) -> SleepSystem:
    """创建睡眠系统"""
    return SleepSystem(
        enable_sleep=enable_sleep,
        enable_dream=enable_dream,
        sleep_threshold=sleep_threshold,
    )


# ══════════════════════════════════════════════════════
# 食欲素/下丘脑分泌素系统 (Orexin/Hypocretin)
# ══════════════════════════════════════════════════════

class OrexinSystem:
    """食欲素系统 — 觉醒促进与失眠关键通路。

    食欲素神经元 (下丘脑外侧) 投射到:
    - 蓝斑 (LC) → NE释放 → 觉醒
    - 中缝核 → 5-HT释放 → 觉醒
    - 结节乳头核 → His释放 → 觉醒
    - 腹侧被盖区 (VTA) → DA释放 → 动机

    失眠机制: orexin过度激活 + GABA不足 → 无法从觉醒切换到睡眠
    Suvorexant (DORA): 阻断orexin1/orexin2 → 降低觉醒驱动 → 入睡

    参考:
    - Sakurai (2007) Nat Rev Neurosci — orexin系统综述
    - Scammell (2015) Neuron — 失眠的神经机制
    """

    def __init__(
        self,
        baseline_orexin: float = 0.5,
        circadian_coupling: float = 0.3,
        gaba_inhibition: float = 0.4,
    ):
        self.baseline = baseline_orexin
        self.circadian_coupling = circadian_coupling
        self.gaba_inhibition = gaba_inhibition
        self.orexin_level = baseline_orexin

    def step(
        self,
        gaba_level: float = 0.5,
        scn_wake_drive: float = 0.5,
        stress_level: float = 0.0,
        receptor_block: float = 0.0,  # suvorexant occupancy (0-1)
    ) -> dict[str, float]:
        """每步更新orexin水平。

        Args:
            gaba_level: GABA抑制强度 (高GABA→抑制orexin)
            scn_wake_drive: SCN觉醒驱动 (白天高, 夜间低)
            stress_level: 应激水平 (应激→激活orexin)
            receptor_block: orexin受体阻断率 (suvorexant)

        Returns:
            orexin_level, effective_orexin (阻断后), arousal_contribution
        """
        # 基础orexin: 昼夜节律调制
        circadian_drive = self.baseline * (0.5 + 0.5 * scn_wake_drive)

        # GABA抑制orexin神经元
        gaba_suppression = gaba_level * self.gaba_inhibition

        # 应激激活orexin (CRH→orexin通路)
        stress_activation = stress_level * 0.3

        # 更新orexin水平
        target = circadian_drive - gaba_suppression + stress_activation
        self.orexin_level = 0.9 * self.orexin_level + 0.1 * max(0.0, min(1.0, target))

        # 受体阻断后的有效orexin
        effective = self.orexin_level * (1.0 - receptor_block)

        # 对觉醒的贡献
        arousal_contribution = effective * 0.3

        return {
            "orexin_level": self.orexin_level,
            "effective_orexin": effective,
            "arousal_contribution": arousal_contribution,
        }


__all__ = [
    'SleepStage',
    'SleepCycle',
    'MemoryReplay',
    'SleepController',
    'MemoryReplayer',
    'SleepSystem',
    'create_sleep_system',
    'OrexinSystem',
]
