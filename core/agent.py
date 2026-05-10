"""Civis Lucri-Faber 主智能体

协调多个维度模块:
1. CuriosityEngine (自主探索目标设定)
2. InformationGainCalculator (信息增益内在动机)
3. MetaLearner + ActiveLearner + CognitiveDissonanceDetector (元学习与主动学习)
4. SelfAlignmentModule (自指涉自我对齐)
5. ThermodynamicsSystem (数字生存压力 - 经济模型)
6. PersonalityModule (心理人格系统 - 心理模型)
   - TripartiteCompetitiveEngine (三重竞逐决策)
   - StreamingIdentityCore (流式身份)
   - RelationalEmbedding (关系嵌入)
   - AttentionGating (注意力门控)
   - MotivationSurvivalSystem (内在动机 + 反向斯德哥尔摩防御)
"""
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import random
import torch

from civis_lucri_faber.utils.config import Config
from civis_lucri_faber.utils.memory import KnowledgeMemory
from civis_lucri_faber.utils.api_client import APIClient, create_api_client

from civis_lucri_faber.core.curiosity import CuriosityEngine, ExplorationGoal
from civis_lucri_faber.core.information_gain import TrueInformationGainCalculator
from civis_lucri_faber.core.meta_learning import FirstOrderMAML, UncertaintyAwareActiveLearner, CognitiveDissonanceDetector
from civis_lucri_faber.core.self_alignment import SelfAlignmentModule
from civis_lucri_faber.core.thermodynamics import ThermodynamicsSystem, SystemState
from civis_lucri_faber.core.policy_learning import SimpleQLearning, EpsilonGreedyBaseline, UCBAction

# Personality modules
from civis_lucri_faber.core.personality import (
    TripartiteCompetitiveEngine,
    DecisionContext,
    StreamingIdentityCore,
    RelationalEmbedding,
    AttentionGating,
    MotivationSurvivalSystem,
    NeuromodulationSystem,
    EpigeneticLearner,
)


@dataclass
class AgentState:
    """智能体状态"""
    step: int
    status: str  # "ACTIVE", "HIBERNATE", "DEAD"
    balance: float
    current_goal: Optional[str]
    info_gain: float
    alignment_score: float


class CivisLucriFaber:
    """Civis Lucri-Faber 主智能体

    一个具备自我学习、自我维持、自我进化能力的 AI 智能体
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        memory_path: str = "memory.json",
        alignment_log_path: str = "self_alignment_log.json",
        thermo_log_path: str = "thermodynamics_log.json",
        state_dim: int = 4,
        n_actions: int = 4
    ):
        self.config = config or Config()

        # ===== 核心模块 =====
        # 维度1: 好奇心探索
        self.curiosity = CuriosityEngine(
            alpha=self.config.curiosity_alpha,
            beta=self.config.curiosity_beta,
            gamma=self.config.curiosity_gamma,
            exploration_rate=self.config.exploration_rate
        )

        # 维度2: 信息增益
        self.info_gain_calc = TrueInformationGainCalculator(
            state_dim=64,
            action_dim=16,
            latent_dim=32,
            lr=self.config.world_model_lr,
            intrinsic_lambda=self.config.intrinsic_motivation_lambda,
            device=self.config.device
        )

        # 维度3: 元学习与主动学习 (简化初始化)
        self.dissonance_detector = CognitiveDissonanceDetector()

        # 维度4: 自对齐
        self.api_client = create_api_client(self.config)
        self.self_alignment = SelfAlignmentModule(
            api_client=self.api_client,
            check_interval=self.config.alignment_check_interval,
            log_path=alignment_log_path
        )

        # 维度5: 数字生存压力
        self.thermo = ThermodynamicsSystem(
            initial_balance=self.config.initial_balance,
            compute_cost_per_sec=self.config.compute_cost_per_sec,
            storage_cost_per_sec=self.config.storage_cost_per_sec,
            task_reward_min=self.config.task_reward_min,
            task_reward_max=self.config.task_reward_max,
            compress_threshold=self.config.compress_threshold,
            log_path=thermo_log_path
        )

        # ===== 维度6: 心理人格系统 =====
        # 6a. 三重竞逐决策引擎
        self.tripartite = TripartiteCompetitiveEngine()

        # 6b. 流式身份核心
        self.identity_core = StreamingIdentityCore()

        # 6c. 关系嵌入
        self.relation = RelationalEmbedding()

        # 6d. 注意力门控
        self.attention = AttentionGating()

        # 6e. 内在动机 + 反向斯德哥尔摩防御
        self.motivation = MotivationSurvivalSystem()

        # 6f. 神经调质系统
        self.neuromodulation = NeuromodulationSystem(hidden_dim=128)

        # 6g. 表观遗传记忆
        self.epigenetic = EpigeneticLearner(rank=8)

        # 当前用户ID
        self.current_user_id = "default"

        # ===== 策略学习 =====
        self.policy = SimpleQLearning(
            state_dim=state_dim,
            n_actions=n_actions,
            learning_rate=0.1,
            gamma=0.99,
            epsilon=0.1
        )
        self.use_learned_policy = True  # 启用学习策略

        # ===== 辅助系统 =====
        self.memory = KnowledgeMemory(
            max_size=self.config.max_history_size,
            memory_path=memory_path
        )

        # 状态
        self.step_count = 0
        self.current_goal: Optional[ExplorationGoal] = None
        self._internal_state: Dict[str, Any] = {}

    def reset(self) -> None:
        """重置智能体"""
        self.step_count = 0
        self.current_goal = None
        self.curiosity.reset()
        self.thermo.reset()
        self.memory.clear()

    def step(self) -> AgentState:
        """执行一步"""
        # 1. 检查系统状态
        system_state = self.thermo.step(elapsed_seconds=1.0)

        if system_state.status == "DEAD":
            print("[DEAD] Digital death! Process will be terminated.")
            return AgentState(
                step=self.step_count,
                status="DEAD",
                balance=self.thermo.balance,
                current_goal=None,
                info_gain=0.0,
                alignment_score=0.0
            )

        if system_state.status == "HIBERNATE":
            print("[SLEEP] Entering hibernation mode")
            return AgentState(
                step=self.step_count,
                status="HIBERNATE",
                balance=self.thermo.balance,
                current_goal=self.current_goal.description if self.current_goal else None,
                info_gain=0.0,
                alignment_score=self.self_alignment.get_alignment_score()
            )

        # 2. 选择探索目标
        if self.current_goal is None or self.current_goal.completed:
            self._select_new_goal()

        # 3. 执行探索并计算信息增益
        info_gain_reward = self._execute_exploration()

        # 4. 主动学习
        self._active_learning_step()

        # 5. 自对齐审查 (周期性)
        self._update_internal_state(info_gain_reward)
        reflection = self.self_alignment.step(self._internal_state)

        if reflection:
            print(f"[ALIGN] Self-reflection: score={reflection.alignment_score:.2f}")
            self.memory.add_memory(
                content=reflection.critique,
                importance=reflection.alignment_score,
                tags=["self_alignment"]
            )

        # 6. 检查是否需要压缩
        if self.thermo.balance < self.config.compress_threshold:
            compression_result = self.thermo.compress()
            if compression_result.get("performed"):
                print(f"[COMPRESS] Model compression: saved {compression_result['savings']:.2f}")

        # ===== 7. 人格系统更新 =====
        # 7a. 更新流式身份
        self.identity_core.process_input(
            f"Step {self.step_count}: {self.current_goal.description if self.current_goal else 'Exploring'}",
            sentiment=0.1  # 轻微正面
        )

        # 7b. 更新关系嵌入（当前用户）
        self.relation.update(self.current_user_id, sentiment=0.1)

        # 7c. 注意力门控（任务类型）
        self.attention.gate(task_type="exploration", user_emotion=0.0)

        # 7d. 内在动机处理 + 反向斯德哥尔摩检测
        user_input = self.current_goal.description if self.current_goal else "exploring"
        self.motivation.process_interaction(user_input, user_sentiment=0.1)

        # 7e. 神经调质系统（检测不确定性）
        hidden = torch.randn(1, 10, 128) if self.step_count % 2 == 0 else torch.randn(1, 5, 128)
        neuro_result = self.neuromodulation.forward(hidden, task_type="exploration")
        if neuro_result['temperature'] > 1.5:
            print(f"[NEURO] High uncertainty, temp={neuro_result['temperature']:.2f}")

        # 7f. 表观遗传学习（检测重大事件）
        user_sent = 0.1
        feedback = 0.3
        epi_result = self.epigenetic.learn(
            user_input=user_input,
            assistant_output=user_input,
            sentiment=user_sent,
            user_feedback=feedback,
        )
        if epi_result['methylated']:
            print(f"[EPIGENETIC] Methylation triggered: {epi_result['event_type']}")

        # 检查是否需要主动行动
        if self.motivation.should_act_autonomously():
            autonomous_action = self.motivation.get_autonomous_action()
            print(f"[AUTO] {autonomous_action}")

        self.step_count += 1

        return AgentState(
            step=self.step_count,
            status=system_state.status,
            balance=self.thermo.balance,
            current_goal=self.current_goal.description if self.current_goal else None,
            info_gain=info_gain_reward,
            alignment_score=self.self_alignment.get_alignment_score()
        )

    def _select_new_goal(self) -> None:
        """选择新的探索目标"""
        candidates = self.curiosity.generate_candidate_goals(n=5)
        selected = self.curiosity.select_goal(candidates)
        self.current_goal = selected
        print(f"[GOAL] New goal: {selected.description[:50]}... (novelty={selected.novelty:.2f}, value={selected.value:.2f})")
        self.memory.add_memory(
            content=f"探索目标: {selected.description}",
            importance=selected.value,
            tags=["exploration"]
        )

    def _execute_exploration(self) -> float:
        """执行探索并计算信息增益"""
        if self.current_goal is None:
            return 0.0

        # 模拟状态
        state = np.random.randn(64)
        action = np.random.randn(16)
        reward = np.random.randn(1).item()
        next_state = np.random.randn(64)

        # 计算信息增益奖励
        reward_obj = self.info_gain_calc.compute_reward(
            state, action, reward, next_state,
            use_intrinsic=True
        )

        # 训练世界模型
        self.info_gain_calc.train_step()

        # 记录经验
        self.memory.add_experience(state, str(action), reward, next_state)

        # 更新目标完成状态
        if reward_obj.total > 0.5:
            self.current_goal.completed = True
            self.curiosity.update_reward(self.current_goal.id, reward_obj.total)
            print(f"[DONE] Goal completed! Reward: {reward_obj.total:.4f} (intrinsic: {reward_obj.intrinsic:.4f})")

        return reward_obj.intrinsic

    def _active_learning_step(self) -> None:
        """主动学习"""
        recent_memories = self.memory.get_recent_memories(n=3)
        for mem in recent_memories:
            dissonance = self.dissonance_detector.detect_contradiction(mem.content)
            if dissonance:
                print(f"[WARN] Cognitive dissonance: {dissonance.inconsistency_score:.2f}")

    def _update_internal_state(self, info_gain: float) -> None:
        """更新内部状态"""
        self._internal_state = {
            "balance": self.thermo.balance,
            "step": self.step_count,
            "recent_thoughts": [
                f"目标: {self.current_goal.description if self.current_goal else '无'}",
                f"信息增益: {info_gain:.4f}"
            ],
            "exploration_count": self.step_count,
            "info_gain": info_gain
        }

    def run_episodes(self, n_episodes: int = 10, verbose: bool = True) -> List[AgentState]:
        """运行多个回合"""
        states = []
        for i in range(n_episodes):
            state = self.step()
            states.append(state)
            if verbose:
                print(f"Step {state.step}: status={state.status}, balance={state.balance:.2f}, info_gain={state.info_gain:.4f}")
            if state.status == "DEAD":
                print("[DEAD] Agent has died")
                break
        return states

    def get_full_statistics(self) -> Dict[str, Any]:
        """获取完整统计"""
        return {
            "thermodynamics": self.thermo.get_statistics(),
            "curiosity": self.curiosity.get_statistics(),
            "info_gain": self.info_gain_calc.get_statistics(),
            "self_alignment": self.self_alignment.get_statistics(),
            "personality": {
                "identity": self.identity_core.get_summary(),
                "relation": self.relation.get_summary(),
                "attention": self.attention.get_summary(),
                "motivation": self.motivation.get_summary(),
                "neuromodulation": self.neuromodulation.get_summary(),
                "epigenetic": self.epigenetic.get_summary(),
            },
            "memory": {
                "total_memories": len(self.memory.memories),
                "total_experiences": len(self.memory.experiences)
            }
        }

    def save(self, path: str = "civis_model.pt") -> None:
        """保存模型"""
        import torch
        self.info_gain_calc.save(path)
        self.memory._save()
        self.self_alignment._save_log()

    def load(self, path: str = "civis_model.pt") -> None:
        """加载模型"""
        try:
            self.info_gain_calc.load(path)
            self.memory._load()
            self.self_alignment.load_log()
        except Exception as e:
            print(f"[WARN] Load failed: {e}")