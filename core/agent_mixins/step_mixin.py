"""Step pipeline mixin — main simulation step, behavior adjustment."""

import numpy as np
import torch
import torch.nn as nn

from core.agent import AgentState


class StepMixin:
    """Mixin providing step() and related simulation-cycle methods.

    Requires the host class to provide:
        _internal_state, _write, _last_step_time, step_count,
        _build_state_vector(), _neural_self_regulation_step(),
        _adjust_behavior_by_internal_state(),
        psychiatric_sim, personality_adapter, memory_adapter, bio_linguistic,
        bus, hippocampus, distributed_memory, memory, config, api_client,
        tools, censor, prefrontal, basal_ganglia, curiosity, dissonance_detector,
        info_gain_calc, tripartite, identity_core, relation, attention,
        current_user_id, limbic, amygdala, ras, brainstem, scn,
        reward_circuit, stress_axis, autonomic, sleep, neuroplasticity,
        interoception, ans_polyvagal, mirror_neuron,
    """

    def step(self, user_input: str = None, user_sentiment: float = 0.0) -> AgentState:
        """单步仿真管道 — 14 脑区 + 10 内稳态 + 4 高级认知

        Pipeline:
        1. 感觉输入 (Censor + 输入向量)
        2. 脑干/自主神经 (Brainstem → ANS → Polyvagal)
        3. 昼夜节律 (SCN → 褪黑素/警觉度)
        4. 边缘系统 (Limbic → Amygdala → RAS)
        5. 内感受/稳态 (Interoception → Stress Axis → Reward Circuit)
        6. 神经递质 (Dopamine/Serotonin/etc.)
        7. 睡眠调控 (Sleep)
        8. 神经可塑性 (Neuroplasticity)
        9. 海马体 (Hippocampus → 分布式记忆)
        10. 前额叶 (PFC → 冲动控制 → 工作记忆)
        11. 基底节 (BG → 策略选择)
        12. 镜像神经元 (Mirror → 共情)
        13. 高级认知 (Curiosity → Dissonance → InfoGain → Tripartite)
        14. 自我认同 (Identity → Relation → Attention)
        15. 内部状态聚合 + 行为调整
        16. 精神疾病模拟
        17. 主动学习 (Active Learner + World Model)
        18. 人格系统
        19. 记忆系统
        20. 生物-语言耦合
        21. 事件发布
        22. 精神病学历程追踪
        """
        import time
        now = time.time()
        dt = now - self._last_step_time if self._last_step_time > 0 else 0.1
        dt = min(dt, 10.0)
        self._last_step_time = now
        self.step_count += 1

        s = self._internal_state

        # ══════════════════════════════════════════════════
        # 1. 感觉输入 — Censor + 外部刺激编码
        # ══════════════════════════════════════════════════
        if user_input is not None:
            s['external_stimulus'] = min(1.0, abs(user_sentiment) + 0.3)
            s['current_input'] = user_input[:200]
        else:
            s['external_stimulus'] = max(0.0, s.get('external_stimulus', 0.3) - 0.05 * dt)
            s['current_input'] = ''

        # ── Censor 扫描 ──
        if user_input is not None and self.censor is not None:
            try:
                censor_result = self.censor(user_input, self._internal_state)
                s['censor_last_result'] = censor_result
            except RuntimeError:
                pass  # Neural module call failed

        # ══════════════════════════════════════════════════
        # 2. 脑干 + 自主神经 (Brainstem → ANS → Polyvagal)
        # ══════════════════════════════════════════════════
        try:
            bsm_state = self.brainstem(self._internal_state)
            s.update(bsm_state)
            self._write('brainstem', bsm_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        try:
            ans_state = self.autonomic(s)
            s.update(ans_state)
            self._write('autonomic', ans_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        try:
            poly_state = self.ans_polyvagal(s)
            s.update(poly_state)
            self._write('polyvagal', poly_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 3. 昼夜节律 (SCN → 褪黑素/警觉度)
        # ══════════════════════════════════════════════════
        try:
            scn_state = self.scn(dt)
            s.update(scn_state)
            self._write('scn', scn_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 4. 边缘系统 (Limbic → Amygdala → RAS)
        # ══════════════════════════════════════════════════
        try:
            limbic_input = torch.FloatTensor([
                s.get('external_stimulus', 0.3),
                s.get('cortisol_level', s.get('hormone_cortisol', 0.3)),
                s.get('nt_dopamine', 0.5),
                s.get('nt_serotonin', 0.5),
                s.get('limbic_valence', 0.0),
                s.get('limbic_arousal', 0.5),
            ]).unsqueeze(0)
            limbic_out = self.limbic(limbic_input)
            limbic_state = {k: float(v) for k, v in limbic_out.items()}
            s.update(limbic_state)
            self._write('limbic', limbic_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        try:
            amygdala_input = torch.FloatTensor([
                s.get('external_stimulus', 0.3),
                s.get('limbic_valence', 0.0),
                s.get('limbic_arousal', 0.5),
                s.get('cortisol_level', s.get('hormone_cortisol', 0.3)),
                user_sentiment,
            ]).unsqueeze(0)
            amygdala_out = self.amygdala(amygdala_input)
            amygdala_state = {k: float(v) for k, v in amygdala_out.items()}
            s.update(amygdala_state)
            self._write('amygdala', amygdala_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        try:
            ras_input = torch.FloatTensor([
                s.get('external_stimulus', 0.3),
                s.get('limbic_arousal', 0.5),
                s.get('scn_alertness', 0.5),
            ]).unsqueeze(0)
            ras_out = self.ras(ras_input)
            ras_state = {k: float(v) for k, v in ras_out.items()}
            s.update(ras_state)
            self._write('ras', ras_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 5. 内感受 / 稳态 (Interoception → Stress → Reward)
        # ══════════════════════════════════════════════════
        try:
            intero_state = self.interoception(s)
            s.update(intero_state)
            self._write('interoception', intero_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        try:
            stress_state = self.stress_axis(s)
            s.update(stress_state)
            self._write('stress_axis', stress_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        try:
            reward_input = torch.FloatTensor([
                s.get('nt_dopamine', 0.5),
                s.get('limbic_valence', 0.0),
                s.get('external_stimulus', 0.3),
            ]).unsqueeze(0)
            reward_out = self.reward_circuit(reward_input)
            reward_state = {k: float(v) for k, v in reward_out.items()}
            s.update(reward_state)
            self._write('reward_circuit', reward_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 6. 神经递质 (Dopamine/Serotonin/etc.)
        # ══════════════════════════════════════════════════
        # ── 多巴胺 ──
        da_base = s.get('nt_dopamine', 0.5)
        reward_signal = s.get('reward_prediction_error', 0.0)
        novelty = s.get('curiosity_novelty', 0.0)
        s['nt_dopamine'] = float(np.clip(
            da_base + reward_signal * 0.1 + novelty * 0.05 - 0.01 * dt, 0.05, 1.0
        ))

        # ── 5-羟色胺 ──
        serotonin_base = s.get('nt_serotonin', 0.5)
        stress = s.get('cortisol_level', s.get('hormone_cortisol', 0.3))
        s['nt_serotonin'] = float(np.clip(
            serotonin_base - stress * 0.05 - 0.005 * dt, 0.05, 1.0
        ))

        # ── 去甲肾上腺素 ──
        ne_base = s.get('nt_norepinephrine', 0.3)
        arousal = s.get('limbic_arousal', 0.5)
        alertness = s.get('scn_alertness', 0.5)
        s['nt_norepinephrine'] = float(np.clip(
            ne_base + arousal * 0.05 + alertness * 0.03 - 0.01 * dt, 0.05, 1.0
        ))

        # ── GABA ──
        gaba_base = s.get('nt_gaba', 0.5)
        s['nt_gaba'] = float(np.clip(
            gaba_base + 0.02 * s.get('nt_serotonin', 0.5) - 0.01 * dt, 0.1, 1.0
        ))

        # ── 乙酰胆碱 ──
        ach_base = s.get('nt_acetylcholine', 0.5)
        s['nt_acetylcholine'] = float(np.clip(
            ach_base + alertness * 0.05 - fatigue * 0.03 if (fatigue := s.get('sleep_fatigue', 0.3)) else ach_base,
            0.05, 1.0,
        ))

        # ── 催产素 ──
        oxytocin_base = s.get('hormone_oxytocin', 0.3)
        touch = s.get('interoception_touch', 0.0)
        s['hormone_oxytocin'] = float(np.clip(
            oxytocin_base + touch * 0.05 + 0.01 * dt, 0.05, 1.0
        ))

        # ══════════════════════════════════════════════════
        # 7. 睡眠调控
        # ══════════════════════════════════════════════════
        try:
            sleep_state = self.sleep(s)
            s.update(sleep_state)
            self._write('sleep', sleep_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 8. 神经可塑性
        # ══════════════════════════════════════════════════
        try:
            plasticity_state = self.neuroplasticity(s)
            s.update(plasticity_state)
            self._write('neuroplasticity', plasticity_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 9. 海马体 (Hippocampus → 分布式记忆)
        # ══════════════════════════════════════════════════
        try:
            state_vec = self._build_state_vector()
            if s.get('external_stimulus', 0.0) > 0.5:
                self.hippocampus.encode_memory(
                    state=state_vec,
                    action=f"step_{self.step_count}",
                    reward=s.get('limbic_valence', 0.0),
                )

            # 分布式记忆
            valence = s.get('limbic_valence', 0.0)
            arousal_val = s.get('limbic_arousal', 0.5)
            importance = max(0.1, abs(valence) * 0.5 + 0.5)
            trace_id = self.distributed_memory.encode(
                state=state_vec,
                valence=valence,
                arousal=arousal_val,
                importance=importance,
            )
            self._write('distributed_memory_trace', trace_id)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 10. 前额叶 (PFC → 冲动控制 → 工作记忆)
        # ══════════════════════════════════════════════════
        try:
            state_t = torch.FloatTensor(self._build_state_vector()).unsqueeze(0)
            pfc_out = self.prefrontal(state_t)
            pfc_state = {k: float(v) for k, v in pfc_out.items()}
            s.update(pfc_state)
            self._write('prefrontal', pfc_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 11. 基底节 (BG → 策略选择)
        # ══════════════════════════════════════════════════
        try:
            bg_out = self.basal_ganglia(state_t)
            bg_state = {k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in bg_out.items()}
            s.update(bg_state)
            self._write('basal_ganglia', bg_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 12. 镜像神经元 (Mirror → 共情)
        # ══════════════════════════════════════════════════
        try:
            mirror_input = torch.FloatTensor([
                user_sentiment,
                s.get('limbic_valence', 0.0),
                s.get('limbic_arousal', 0.5),
                s.get('hormone_oxytocin', 0.3),
            ]).unsqueeze(0)
            mirror_out = self.mirror_neuron(mirror_input)
            mirror_state = {k: float(v) for k, v in mirror_out.items()}
            s.update(mirror_state)
            self._write('mirror_neuron', mirror_state)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 13. 高级认知 (Curiosity → Dissonance → InfoGain → Tripartite)
        # ══════════════════════════════════════════════════
        try:
            curiosity_out = self.curiosity(s)
            s.update(curiosity_out)
            self._write('curiosity', curiosity_out)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        try:
            dissonance_out = self.dissonance_detector.detect(s)
            if isinstance(dissonance_out, dict):
                s.update(dissonance_out)
                self._write('dissonance', dissonance_out)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        try:
            info_gain_out = self.info_gain_calc.compute(s)
            s.update(info_gain_out)
            self._write('info_gain', info_gain_out)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        try:
            tripartite_out = self.tripartite(s)
            s.update(tripartite_out)
            self._write('tripartite', tripartite_out)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 14. 自我认同 (Identity → Relation → Attention)
        # ══════════════════════════════════════════════════
        try:
            identity_out = self.identity_core(s)
            s.update(identity_out)
            self._write('identity_core', identity_out)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        try:
            relation_out = self.relation(s)
            s.update(relation_out)
            self._write('relation', relation_out)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        try:
            attention_out = self.attention(s)
            s.update(attention_out)
            self._write('attention', attention_out)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 15. 内部状态聚合 + 行为调整
        # ══════════════════════════════════════════════════
        self._neural_self_regulation_step()
        self._adjust_behavior_by_internal_state()

        # ══════════════════════════════════════════════════
        # 16. 精神疾病模拟
        # ══════════════════════════════════════════════════
        try:
            psych_out = self.psychiatric_sim(s)
            s.update(psych_out)
            self._write('psychiatric_sim', psych_out)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 17. 主动学习 (Active Learner + World Model)
        # ══════════════════════════════════════════════════
        try:
            # Active Learner 不确定性更新
            novelty_signal = s.get('curiosity_novelty', 0.0)
            s['active_learner_uncertainty'] = float(np.clip(
                s.get('active_learner_uncertainty', 0.5) + novelty_signal * 0.1 - 0.02 * dt,
                0.0, 1.0,
            ))
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 18. 人格系统
        # ══════════════════════════════════════════════════
        if self.personality_adapter is not None:
            try:
                self.personality_adapter.update_from_personality(
                    tripartite=self.tripartite,
                    identity_core=self.identity_core,
                    relation=self.relation,
                    attention=self.attention,
                )
            except RuntimeError:
                pass  # Neural module call failed

        # ══════════════════════════════════════════════════
        # 19. 记忆系统
        # ══════════════════════════════════════════════════
        if self.memory_adapter is not None:
            try:
                recent_memories = self.memory.get_recent_memories(n=5)
                memory_data = []
                for mem in recent_memories:
                    memory_data.append({
                        'content': mem.content[:100] if hasattr(mem, 'content') else str(mem)[:100],
                        'valence': float(getattr(mem, 'valence', 0.0)),
                        'arousal': float(getattr(mem, 'arousal', 0.5)),
                        'importance': float(getattr(mem, 'importance', 0.5)),
                        'emotion_tag': getattr(mem, 'emotion_tag', 'neutral'),
                        'source': getattr(mem, 'source', ''),
                    })
                if memory_data:
                    bio_for_memory = {
                        'valence': float(s.get('limbic_valence', 0.0)),
                        'arousal': float(s.get('limbic_arousal', 0.5)),
                        'cortisol': float(s.get('cortisol_level', s.get('hormone_cortisol', 0.3))),
                        'serotonin': float(s.get('nt_serotonin', 0.5)),
                    }
                    self.memory_adapter.process_memories(
                        memory_data,
                        current_bio_state=bio_for_memory,
                        user_id=self.current_user_id,
                    )
            except RuntimeError:
                pass  # Neural module call failed

        # ══════════════════════════════════════════════════
        # 20. 生物-语言耦合
        # ══════════════════════════════════════════════════
        if self.bio_linguistic is not None:
            try:
                bio_state_for_lang = {
                    'dopamine': float(s.get('nt_dopamine', 0.5)),
                    'serotonin': float(s.get('nt_serotonin', 0.5)),
                    'norepinephrine': float(s.get('nt_norepinephrine', 0.3)),
                    'cortisol': float(s.get('cortisol_level', s.get('hormone_cortisol', 0.3))),
                    'fatigue': float(s.get('sleep_fatigue', 0.3)),
                    'arousal': float(s.get('limbic_arousal', 0.5)),
                    'valence': float(s.get('limbic_valence', 0.0)),
                    'oxytocin': float(s.get('hormone_oxytocin', 0.3)),
                    'emotion': s.get('limbic_emotion', 'neutral'),
                    'heart_rate': float(s.get('bsm_heart_rate', 72.0)),
                    'respiratory_rate': float(s.get('bsm_respiratory_rate', 12.0)),
                    'defense': s.get('bsm_defense_behavior', ''),
                }
                self.bio_linguistic.update(bio_state_for_lang)
            except RuntimeError:
                pass  # Neural module call failed

        # ══════════════════════════════════════════════════
        # 21. 事件发布
        # ══════════════════════════════════════════════════
        try:
            from core.events import NEURAL_STEP_COMPLETED
            self.bus.publish(NEURAL_STEP_COMPLETED, {
                "step": self.step_count,
                "internal_state": dict(s),
                "dt": dt,
            }, source="agent")
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ══════════════════════════════════════════════════
        # 22. 精神病学历程追踪
        # ══════════════════════════════════════════════════
        try:
            self.psychiatric_sim.track_trajectory(s)
        except RuntimeError:
            pass  # Neural module call failed (PyTorch or attribute error)

        # ── 生成 AgentState ──
        status = "ALIVE"
        energy = s.get('energy', 1.0)
        if energy < 0.05:
            status = "DEAD"
        elif s.get('sleep_stage', 'awake') == 'deep':
            status = "HIBERNATE"

        return AgentState(
            status=status,
            step=self.step_count,
            emotion=s.get('limbic_emotion', 'neutral'),
            arousal=s.get('limbic_arousal', 0.5),
            valence=s.get('limbic_valence', 0.0),
            energy=energy,
            internal_state=dict(s),
        )

    def _adjust_behavior_by_internal_state(self):
        """根据内部状态调整行为参数（策略覆盖 + 内稳态补偿）

        补偿机制:
        - 体温过低 → 战栗产热（提高代谢率 + 心率）
        - 体温过高 → 出汗散热（提高心率，降低外周阻力）
        - 脱水 → 抗利尿（降低排尿率，提高口渴感）
        - 低血糖 → 糖异生（提高皮质醇 + 胰高血糖素）
        - 缺氧 → 过度通气（提高呼吸频率）
        - 高 CO₂ → 呼吸驱动增加
        - 低钠 → ADH 释放
        - 高钾 → 胰岛素释放
        - 酸中毒 → 过度通气代偿
        - 碱中毒 → 低通气代偿

        行为覆盖:
        - fight → PFC 抑制降低 + NE 激增
        - flight → 皮质醇激增 + 运动准备
        - freeze → PFC 过度抑制 + 迷走激活
        """
        s = self._internal_state

        # ── 体温调节补偿 ──
        temp = s.get('bsm_body_temperature', 37.0)
        if temp < 36.0:
            # 低温 → 战栗产热
            s['bsm_heart_rate'] = float(s.get('bsm_heart_rate', 72.0)) + 15.0
            s['bsm_metabolic_rate'] = float(s.get('bsm_metabolic_rate', 1.0)) + 0.5
            s['shivering'] = True
        elif temp > 38.5:
            # 高温 → 出汗散热
            s['bsm_heart_rate'] = float(s.get('bsm_heart_rate', 72.0)) + 10.0
            s['bsm_metabolic_rate'] = max(0.5, float(s.get('bsm_metabolic_rate', 1.0)) - 0.2)
            s['sweating'] = True
        else:
            s['shivering'] = False
            s['sweating'] = False

        # ── 脱水补偿 ──
        hydration = s.get('bsm_hydration', 1.0)
        if hydration < 0.8:
            s['thirst_level'] = min(1.0, (1.0 - hydration) * 2.0)
            s['bsm_urine_output'] = max(0.1, float(s.get('bsm_urine_output', 1.0)) * hydration)
        else:
            s['thirst_level'] = max(0.0, float(s.get('thirst_level', 0.0)) - 0.01)

        # ── 低血糖补偿 ──
        glucose = s.get('bsm_blood_glucose', 90.0)
        if glucose < 70.0:
            s['cortisol_level'] = float(s.get('cortisol_level', s.get('hormone_cortisol', 0.3))) + 0.15
            s['hormone_glucagon'] = float(s.get('hormone_glucagon', 0.3)) + 0.2
            s['hypoglycemia_symptoms'] = True
        else:
            s['hypoglycemia_symptoms'] = False

        # ── 缺氧/高CO₂补偿 ──
        spo2 = s.get('bsm_spo2', 98.0)
        if spo2 < 92.0:
            s['bsm_respiratory_rate'] = float(s.get('bsm_respiratory_rate', 12.0)) + 8.0
            s['hypoxia_flag'] = True
        else:
            s['hypoxia_flag'] = False

        co2 = s.get('bsm_blood_co2', 40.0)
        if co2 > 50.0:
            s['bsm_respiratory_rate'] = float(s.get('bsm_respiratory_rate', 12.0)) + 5.0
        elif co2 < 30.0:
            # 低 CO₂ → 低通气
            s['bsm_respiratory_rate'] = max(6.0, float(s.get('bsm_respiratory_rate', 12.0)) - 3.0)

        # ── 电解质平衡 ──
        sodium = s.get('bsm_blood_sodium', 140.0)
        if sodium < 135.0:
            s['hormone_adh'] = float(s.get('hormone_adh', 0.3)) + 0.3
        elif sodium > 145.0:
            s['hormone_adh'] = max(0.0, float(s.get('hormone_adh', 0.3)) - 0.2)

        potassium = s.get('bsm_blood_potassium', 4.0)
        if potassium > 5.5:
            s['hormone_insulin'] = float(s.get('hormone_insulin', 0.3)) + 0.2

        # ── 酸碱平衡 ──
        ph = s.get('bsm_blood_ph', 7.4)
        if ph < 7.35:
            # 酸中毒 → 过度通气代偿
            s['bsm_respiratory_rate'] = float(s.get('bsm_respiratory_rate', 12.0)) + 6.0
            s['acidosis_flag'] = True
        elif ph > 7.45:
            # 碱中毒 → 低通气代偿
            s['bsm_respiratory_rate'] = max(6.0, float(s.get('bsm_respiratory_rate', 12.0)) - 4.0)
            s['alkalosis_flag'] = True
        else:
            s['acidosis_flag'] = False
            s['alkalosis_flag'] = False

        # ── 防御行为覆盖 ──
        defense = s.get('bsm_defense_behavior', '')
        if defense == 'fight':
            s['pfc_inhibition'] = max(0.0, float(s.get('pfc_inhibition', 0.5)) - 0.3)
            s['nt_norepinephrine'] = min(1.0, float(s.get('nt_norepinephrine', 0.3)) + 0.4)
            s['bsm_heart_rate'] = float(s.get('bsm_heart_rate', 72.0)) + 30.0
            s['bsm_blood_pressure_systolic'] = float(s.get('bsm_blood_pressure_systolic', 120.0)) + 20.0
        elif defense == 'flight':
            s['cortisol_level'] = min(1.0, float(s.get('cortisol_level', s.get('hormone_cortisol', 0.3))) + 0.3)
            s['bsm_heart_rate'] = float(s.get('bsm_heart_rate', 72.0)) + 25.0
            s['limbic_arousal'] = min(1.0, float(s.get('limbic_arousal', 0.5)) + 0.3)
        elif defense == 'freeze':
            s['pfc_inhibition'] = min(1.0, float(s.get('pfc_inhibition', 0.5)) + 0.4)
            s['ans_polyvagal_state'] = 'dorsal_vagal'
            s['limbic_arousal'] = max(0.0, float(s.get('limbic_arousal', 0.5)) - 0.2)
            s['bsm_heart_rate'] = max(40.0, float(s.get('bsm_heart_rate', 72.0)) - 15.0)

        # ── 能量代谢 ──
        energy = s.get('energy', 1.0)
        if energy < 0.3:
            s['limbic_arousal'] = min(float(s.get('limbic_arousal', 0.5)), 0.4)
            s['scn_alertness'] = min(float(s.get('scn_alertness', 0.5)), 0.4)
            s['nt_dopamine'] = max(0.1, float(s.get('nt_dopamine', 0.5)) - 0.15)

        # ── HRV 与自主平衡 ──
        hrv = s.get('ans_hrv', 0.6)
        if hrv < 0.3:
            # 低 HRV → 交感主导 → 压力标记
            s['sympathetic_dominance'] = True
            s['cortisol_level'] = min(1.0, float(s.get('cortisol_level', s.get('hormone_cortisol', 0.3))) + 0.05)

        # ── 稳态超载 ──
        allostatic_load = s.get('allostatic_load', 0.0)
        if allostatic_load > 0.7:
            # 高稳态超载 → 多系统衰退
            s['pfc_inhibition'] = min(1.0, float(s.get('pfc_inhibition', 0.5)) + 0.1)
            s['nt_dopamine'] = max(0.1, float(s.get('nt_dopamine', 0.5)) - 0.05)
            s['nt_serotonin'] = max(0.1, float(s.get('nt_serotonin', 0.5)) - 0.05)
            s['brain_health'] = max(0.0, float(s.get('brain_health', 0.8)) - 0.01)
