"""StateMixin — State aggregation, statistics, save/load, episode running

从 monolithic agent.py 中提取的状态相关方法:
- get_full_statistics()
- save(), load()
- run_episodes()
"""

from typing import Any

from core.agent import AgentState


class StateMixin:
    """状态管理 Mixin — 统计聚合、保存/加载、回合运行

    使用 self 访问 Simulacrum 实例的属性:
    - self._internal_state: 内部状态字典
    - self.advanced_emotion: 高级情绪系统
    - self.ans: 自主神经系统
    - self.hpa_axis: HPA轴
    - self.glial: 胶质系统
    - self.allostatic: 稳态调节
    - self.predictive_coding: 预测编码
    - self.social_cognition: 社会认知
    - self.self_awareness: 自我意识
    - self.thermo: 热力学系统
    - self.curiosity: 好奇心引擎
    - self.info_gain_calc: 信息增益计算器
    - self.self_alignment: 自对齐模块
    - self.neural_pruning: 神经修剪系统
    - self.basal_ganglia: 基底神经节
    - self.neurotransmitter: 神经递质系统
    - self.neuroplasticity: 神经可塑性
    - self.prefrontal: 前额叶
    - self.angular_gyrus: 角回
    - self.hormones: 激素系统
    - self.brainstem: 脑干
    - self.scn: 视交叉上核
    - self.limbic: 边缘系统
    - self.hippocampus: 海马体
    - self.current_mood: 当前心境状态
    - self.distributed_memory: 分布式记忆
    - self.psychotherapy: 心理治疗系统
    - self.rhythm: 节律系统
    - self.sleep_system: 睡眠系统
    - self.identity_core: 身份核心
    - self.relation: 关系嵌入
    - self.attention: 注意门控
    - self.motivation: 动机系统
    - self.neuromodulation: 神经调质
    - self.epigenetic: 表观遗传
    - self.bus: 事件总线
    - self.memory: 知识记忆
    - self.hw: 硬件生命体征
    - self.step_count: 步数计数
    - self.config: 配置
    """

    def run_episodes(self, n_episodes: int = 10, verbose: bool = True) -> list[AgentState]:
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

    def get_full_statistics(self) -> dict[str, Any]:
        """获取完整统计"""
        adv_emotion_stats = {}
        if self.advanced_emotion is not None:
            adv_emotion_stats = self.advanced_emotion.get_summary()

        self_regulation_stats = {}
        for name, getter in [
            ("autonomic_nervous_system", lambda: self.ans.get_summary()),
            ("hpa_axis", lambda: self.hpa_axis.get_summary()),
            ("glial_system", lambda: self.glial.get_summary()),
            ("allostatic_regulation", lambda: self.allostatic.get_summary()),
            ("predictive_coding", lambda: self.predictive_coding.get_summary()),
            ("social_cognition", lambda: self.social_cognition.get_summary()),
            ("self_awareness", lambda: self.self_awareness.get_summary()),
        ]:
            try:
                self_regulation_stats[name] = getter()
            except Exception:
                self_regulation_stats[name] = {"error": "unavailable"}

        return {
            "thermodynamics": self.thermo.get_statistics(),
            "curiosity": self.curiosity.get_statistics(),
            "info_gain": self.info_gain_calc.get_statistics(),
            "active_learner": {
                "uncertainty": self._internal_state.get('active_learner_uncertainty', 0.0),
            },
            "self_alignment": self.self_alignment.get_statistics(),
            "advanced_emotion": adv_emotion_stats,
            "neural_pruning": self.neural_pruning.get_summary(),
            "neural_self_regulation": self_regulation_stats,
            "brain_regions": {
                "basal_ganglia": {
                    "habit_strength": self._internal_state.get('bg_habit_strength', 0.0),
                    "td_error": self._internal_state.get('bg_td_error', 0.0),
                },
                "neurotransmitter": {
                    "dopamine": self._internal_state.get('nt_dopamine', 0.5),
                    "serotonin": self._internal_state.get('nt_serotonin', 0.5),
                    "state": self._internal_state.get('nt_state', 'neutral'),
                },
                "neuroplasticity": {
                    "bdnf": self._internal_state.get('plasticity_bdnf', 0.5),
                    "active_synapses": self._internal_state.get('plasticity_synapses', 0),
                },
                "prefrontal_cortex": {
                    "maturity": self._internal_state.get('pfc_maturity', 0.0),
                    "inhibition_gate": self._internal_state.get('pfc_inhibition', 0.0),
                    "plan_depth": self._internal_state.get('pfc_plan_depth', 1),
                    "overrode_bg": self._internal_state.get('pfc_overrode_bg', False),
                },
                "angular_gyrus": {
                    "scene": self._internal_state.get('ag_scene', 'neutral'),
                    "n_modalities": self._internal_state.get('ag_n_present', 0),
                    "predictions": self._internal_state.get('ag_n_predicted', 0),
                },
                "hormones": self.hormones.get_summary(),
                "brainstem": {
                    "arousal": self._internal_state.get('bsm_arousal', 0.5),
                    "arousal_name": self._internal_state.get('bsm_arousal_name', 'RELAXED'),
                    "consciousness_gate": self._internal_state.get('bsm_consciousness_gate', 0.5),
                    "respiratory_rate": round(self._internal_state.get('bsm_respiratory_rate', 12.0), 1),
                    "heart_rate": round(self._internal_state.get('bsm_heart_rate', 72.0), 1),
                    "blood_pressure": round(self._internal_state.get('bsm_blood_pressure', 120.0), 1),
                    "defense_behavior": self._internal_state.get('bsm_defense_behavior', 'freeze'),
                    "pain_gating": round(self._internal_state.get('bsm_pain_gating', 0.5), 3),
                },
                "scn": {
                    "circadian_hour": round(self._internal_state.get('scn_circadian_hour', 0), 2),
                    "melatonin": round(self._internal_state.get('scn_melatonin', 0), 4),
                    "wake_drive": round(self._internal_state.get('scn_wake_drive', 0.5), 4),
                    "sleep_pressure": round(self._internal_state.get('scn_sleep_pressure', 0), 4),
                    "alertness": round(self._internal_state.get('scn_alertness', 0.5), 4),
                    "temperature": round(self._internal_state.get('scn_temperature', 37.0), 2),
                },
                "limbic": {
                    "emotion": self._internal_state.get('limbic_emotion', 'neutral'),
                    "valence": round(self._internal_state.get('limbic_valence', 0), 3),
                    "arousal": round(self._internal_state.get('limbic_arousal', 0), 3),
                    "response": self._internal_state.get('limbic_response', 'calm'),
                },
                "hippocampus": self.hippocampus.get_summary(),
                "mood_system": {
                    "valence": self.current_mood.valence if hasattr(self, 'current_mood') else 0.0,
                    "arousal": self.current_mood.arousal if hasattr(self, 'current_mood') else 0.5,
                    "dominance": self.current_mood.dominance if hasattr(self, 'current_mood') else 0.5,
                },
                "distributed_memory": self.distributed_memory.get_summary(),
                "emotion_extensions": {
                    "emergent_emotion": self._internal_state.get('emergent_emotion', []),
                    "emotion_dynamics_criticality": self._internal_state.get('emotion_dynamics_criticality', 0.0),
                    "emotion_regulation_strategy": self._internal_state.get('emotion_regulation_strategy', 'none'),
                    "dominant_social_emotion": self._internal_state.get('dominant_social_emotion', 'neutral'),
                },
                "cognitive_extensions": {
                    "crossmodal_coherence": self._internal_state.get('crossmodal', {}).get('crossmodal_coherence', 0.5),
                    "nm_dopamine": self._internal_state.get('nm_dopamine', 0.5),
                    "nm_serotonin": self._internal_state.get('nm_serotonin', 0.5),
                },
                "therapy": {
                    "psychotherapy_active": bool(self.psychotherapy.active_treatments) if hasattr(self.psychotherapy, 'active_treatments') else False,
                    "psychometric_available": True,
                },
                "rhythm": self.rhythm.get_summary() if hasattr(self, 'rhythm') else {},
                "sleep": self.sleep_system.get_summary(),
            },
            "personality": {
                "identity": self.identity_core.get_summary(),
                "relation": self.relation.get_summary(),
                "attention": self.attention.get_summary(),
                "motivation": self.motivation.get_summary(),
                "neuromodulation": self.neuromodulation.get_summary(),
                "epigenetic": self.epigenetic.get_summary(),
            },
            "event_bus": self.bus.get_stats(),
            "memory": {
                "total_memories": len(self.memory.memories),
                "total_experiences": len(self.memory.experiences)
            },
            "hardware_vitals": self.hw.get_bilingual_summary(),
        }

    def save(self, path: str = "civis_model.pt") -> None:
        """保存模型"""
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