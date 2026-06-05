"""Chat interface mixin — LLM interaction, cognitive gating, bio-prompt construction."""

import re

import numpy as np
import torch

from core.types import ChatResponse


class ChatMixin:
    """Mixin providing chat() and related LLM interaction methods.

    Requires the host class to provide:
        _internal_state, _write, _chat_history, _max_chat_history,
        step(), _build_state_vector(), _cognitive_pre_gate(),
        _gate_fallback(), _compute_llm_params(), _apply_strategy_to_params(),
        _build_bio_prompt_with_strategy(), _retrieve_rag_context(),
        _build_chat_messages(), _llm_tool_loop(), _cognitive_post_filter(),
        _apply_post_filter(), _proactive_learning(), _prepare_vocalization_input(),
        _get_emotion_vector(), _get_censor_result_dict(),
        psychiatric_sim, personality_adapter, memory_adapter, bio_linguistic,
        bus, hippocampus, distributed_memory, memory, config, api_client,
        tools, censor, prefrontal, basal_ganglia, curiosity, dissonance_detector,
        info_gain_calc, tripartite, identity_core, relation, attention,
        current_user_id, _STRATEGY_MODS
    """

    # ---- Strategy modifier table (enhanced: more aggressive parameter shifts) ----
    _STRATEGY_MODS = {
        "explore":  {"temp_factor": 1.4, "max_tok_factor": 1.0,  "presence_adj": -0.2, "hint": "自由探索，可以发散思维，展现好奇心"},
        "concise":  {"temp_factor": 0.5, "max_tok_factor": 0.2,  "presence_adj": 0.3,  "hint": "简洁回答，不要展开，直击要点"},
        "wait":     {"temp_factor": 0.4, "max_tok_factor": 0.4,  "presence_adj": 0.1,  "hint": "先思考再回答，可以反问用户澄清问题"},
        "refuse":   {"temp_factor": 0.3, "max_tok_factor": 0.08, "presence_adj": 0.5,  "hint": "婉拒回答这个话题，礼貌地转移"},
        "suppress": {"temp_factor": 0.25,"max_tok_factor": 0.1,  "presence_adj": 0.6,  "hint": "极度克制，只说必要的话，字数尽量少"},
        "burst":    {"temp_factor": 1.8, "max_tok_factor": 1.0,  "presence_adj": -0.4, "hint": "情绪爆发！激烈、直接、不加修饰地表达！"},
        "avoid":    {"temp_factor": 0.4, "max_tok_factor": 0.2,  "presence_adj": 0.3,  "hint": "回避这个话题，礼貌地转向其他内容"},
    }

    def chat(self, user_input: str, user_sentiment: float = 0.0, video: torch.Tensor = None, condition: str = None, severity: str = "moderate") -> ChatResponse:
        """对话入口 — 三层认知管道

        第一层 (Pre-Gate): PFC 抑制 + BG 策略选择 + RAS 意识门
        第二层 (LLM):      神经递质参数 + RAG + 策略指令
        第三层 (Post-Filter): 预测编码 + 自我认同 + 情绪调节

        Args:
            user_input: 用户输入文本
            user_sentiment: 用户情感 [-1, 1]
            video: 视频帧张量 (B, 3, T, H, W)，供 Censor 微表情分析
            condition: 精神疾病 ID (如 "MDD", "GAD")，应用 profile
            severity: "mild" / "moderate" / "severe"
        """
        # ── 注入 Censor 视频输入 ──
        if video is not None:
            self._write('censor_video', video)

        # ── 注入精神疾病条件 ──
        if condition is not None:
            self.psychiatric_sim.apply_condition(condition, severity=severity)

        # ── step() 更新所有神经模块 ──
        agent_state = self.step(user_input=user_input, user_sentiment=user_sentiment)

        if agent_state.status in ("DEAD", "HIBERNATE"):
            fallback = "我的能量耗尽了..." if agent_state.status == "DEAD" else "我需要休息一下..."
            return ChatResponse(text=fallback, emotion="sadness", arousal=0.1, valence=-0.5,
                                internal_state=dict(self._internal_state), tool_calls=[])

        # ══════════════════════════════════════════════════
        # 第一层：Pre-LLM 认知门控
        # ══════════════════════════════════════════════════
        pre_gate = self._cognitive_pre_gate(user_input)
        if not pre_gate["gate"]:
            # 门控关闭：大脑决定不回答
            gate_text = self._gate_fallback(pre_gate)
            return ChatResponse(
                text=gate_text,
                emotion=self._internal_state.get('limbic_emotion', 'neutral'),
                arousal=self._internal_state.get('limbic_arousal', 0.5),
                valence=self._internal_state.get('limbic_valence', 0.0),
                internal_state=dict(self._internal_state),
                tool_calls=[],
                cognitive_gate=pre_gate,
            )

        # ══════════════════════════════════════════════════
        # 第二层：LLM 生成（策略 + 神经递质 + RAG + 人格 + 记忆）
        # ══════════════════════════════════════════════════
        llm_params = self._compute_llm_params()
        llm_params = self._apply_strategy_to_params(llm_params, pre_gate["strategy"])

        # ── 人格系统更新风格 ──
        personality_style_prompt = ""
        if self.personality_adapter is not None:
            try:
                self.personality_adapter.update_from_personality(
                    tripartite=self.tripartite,
                    identity_core=self.identity_core,
                    relation=self.relation,
                    attention=self.attention,
                )
                personality_style_prompt = self.personality_adapter.generate_style_prompt()
            except Exception:
                pass

        # ── 记忆系统影响风格 ──
        memory_style_prompt = ""
        if self.memory_adapter is not None:
            try:
                # 获取最近的记忆用于风格调制
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
                        'valence': float(self._internal_state.get('limbic_valence', 0.0)),
                        'arousal': float(self._internal_state.get('limbic_arousal', 0.5)),
                        'cortisol': float(self._internal_state.get('cortisol_level', 0.3)),
                        'serotonin': float(self._internal_state.get('nt_serotonin', 0.5)),
                    }
                    self.memory_adapter.process_memories(
                        memory_data,
                        current_bio_state=bio_for_memory,
                        user_id=self.current_user_id,
                    )
                    memory_style_prompt = self.memory_adapter.generate_memory_prompt()
            except Exception:
                pass

        bio_prompt = self._build_bio_prompt_with_strategy(
            pre_gate,
            personality_prompt=personality_style_prompt,
            memory_prompt=memory_style_prompt,
        )
        rag_context = self._retrieve_rag_context(user_input)
        messages = self._build_chat_messages(user_input, bio_prompt, rag_context)

        response_text, tool_calls = self._llm_tool_loop(messages, llm_params)

        # ══════════════════════════════════════════════════
        # 第三层：Post-LLM 质量过滤
        # ══════════════════════════════════════════════════
        post_filter = self._cognitive_post_filter(response_text, user_input)
        response_text = self._apply_post_filter(response_text, messages, post_filter, llm_params)

        # ── 记录对话历史 ──
        self._chat_history.append({"role": "user", "content": user_input})
        self._chat_history.append({"role": "assistant", "content": response_text})
        if len(self._chat_history) > self._max_chat_history * 2:
            self._chat_history = self._chat_history[-(self._max_chat_history * 2):]

        # ══════════════════════════════════════════════════
        # 主动学习：从对话中学习
        # ══════════════════════════════════════════════════
        learning_active = self._proactive_learning(user_input, response_text)

        # ── 编码到海马体 ──
        try:
            state_vec = self._build_state_vector()
            self.hippocampus.encode_memory(state=state_vec, action=f"chat: {user_input[:50]}", reward=0.5)

            # ── 分布式记忆编码 (跨脑区存储) ──
            valence = self._internal_state.get('limbic_valence', 0.0)
            arousal = self._internal_state.get('limbic_arousal', 0.5)
            importance = max(0.1, abs(valence) * 0.5 + 0.5)
            trace_id = self.distributed_memory.encode(
                state=state_vec,
                valence=valence,
                arousal=arousal,
                importance=importance,
            )
            self._write('distributed_memory_trace', trace_id)
        except (RuntimeError, AttributeError):
            pass  # Personality adapter call failed

        # ── 对话触发发声（LLM回复 → 生物-语言耦合 → 发音系统） ──
        vocal_output = None
        try:
            # ── 生物-语言耦合：神经系统直接调制语言输出 ──
            if self.bio_linguistic is not None:
                bio_state_for_lang = {
                    'dopamine': float(self._internal_state.get('nt_dopamine', 0.5)),
                    'serotonin': float(self._internal_state.get('nt_serotonin', 0.5)),
                    'norepinephrine': float(self._internal_state.get('nt_norepinephrine', 0.3)),
                    'cortisol': float(self._internal_state.get('cortisol_level',
                        self._internal_state.get('hormone_cortisol', 0.3))),
                    'fatigue': float(self._internal_state.get('sleep_fatigue', 0.3)),
                    'arousal': float(self._internal_state.get('limbic_arousal', 0.5)),
                    'valence': float(self._internal_state.get('limbic_valence', 0.0)),
                    'oxytocin': float(self._internal_state.get('hormone_oxytocin', 0.3)),
                    'emotion': self._internal_state.get('limbic_emotion', 'neutral'),
                    'heart_rate': float(self._internal_state.get('bsm_heart_rate', 72.0)),
                    'respiratory_rate': float(self._internal_state.get('bsm_respiratory_rate', 12.0)),
                    'defense': self._internal_state.get('bsm_defense_behavior', ''),
                }
                response_text = self.bio_linguistic.process(response_text, bio_state_for_lang)

            # ── 人格系统后处理 ──
            if self.personality_adapter is not None:
                try:
                    response_text = self.personality_adapter.apply_to_text(response_text)
                except Exception:
                    pass

            # ── 记忆系统后处理 ──
            if self.memory_adapter is not None:
                try:
                    response_text = self.memory_adapter.apply_to_text(response_text)
                except Exception:
                    pass

            vocal_indices = self._prepare_vocalization_input(response_text=response_text)
            if vocal_indices is not None:
                from core.events import VOCALIZATION_CONTROL
                respiratory_rate = float(self._internal_state.get('bsm_respiratory_rate', 12.0))
                respiratory_phase = float(self._internal_state.get('bsm_respiratory_phase', 0.5))
                arousal = float(self._internal_state.get('limbic_arousal', 0.5))
                emotion_vec = self._get_emotion_vector()

                vocal_result = self.bus.publish(VOCALIZATION_CONTROL, {
                    "phoneme_indices": vocal_indices,
                    "respiratory_rate": respiratory_rate,
                    "respiratory_phase": respiratory_phase,
                    "emotion_vector": emotion_vec,
                    "arousal": arousal,
                    "internal_state": self._internal_state,
                    "bio_state": {
                        'arousal': arousal,
                        'fatigue': float(self._internal_state.get('sleep_fatigue', 0.3)),
                        'dopamine': float(self._internal_state.get('nt_dopamine', 0.5)),
                        'serotonin': float(self._internal_state.get('nt_serotonin', 0.5)),
                        'norepinephrine': float(self._internal_state.get('nt_norepinephrine', 0.3)),
                        'cortisol': float(self._internal_state.get('cortisol_level',
                            self._internal_state.get('hormone_cortisol', 0.3))),
                        'valence': float(self._internal_state.get('limbic_valence', 0.0)),
                        'heart_rate': float(self._internal_state.get('bsm_heart_rate', 72.0)),
                        'respiratory_rate': respiratory_rate,
                        'emotion': self._internal_state.get('limbic_emotion', 'neutral'),
                    },
                }, source="agent")
                if vocal_result.get('is_speaking'):
                    vocal_output = vocal_result
                    self._last_vocalization = vocal_result
        except Exception:
            import traceback
            traceback.print_exc()

        return ChatResponse(
            emotion=self._internal_state.get('limbic_emotion', 'neutral'),
            arousal=self._internal_state.get('limbic_arousal', 0.5),
            valence=self._internal_state.get('limbic_valence', 0.0),
            internal_state=dict(self._internal_state),
            tool_calls=tool_calls,
            llm_params=llm_params,
            cognitive_gate=pre_gate,
            quality_filter=post_filter,
            learning_active=learning_active,
            vocalization=vocal_output,
            censor_result=self._get_censor_result_dict(),
        )

    # ── 第一层：Pre-LLM 认知门控 ──

    def _cognitive_pre_gate(self, user_input: str) -> dict:
        """大脑在 LLM 之前的认知门控（增强版：对中等状态更敏感）

        PFC 抑制门 + BG 策略选择 + RAS 意识门 + 防御行为
        策略触发阈值降低，中等情绪波动也能选择策略
        """
        s = self._internal_state

        # 1. RAS 意识门 — 不清醒则拒绝
        consciousness = s.get('bsm_consciousness_gate', 0.5)
        if consciousness < 0.25:  # 放宽阈值（原: <0.2）
            return {"strategy": "unconscious", "gate": False, "reason": "意识门关闭",
                    "consciousness": consciousness}

        # 2. 防御行为直接覆盖
        defense = s.get('bsm_defense_behavior', '')
        if defense == 'freeze':
            return {"strategy": "freeze", "gate": False, "reason": "冻结反应"}
        if defense == 'flight':
            return {"strategy": "avoid", "gate": True, "reason": "回避话题",
                    "consciousness": consciousness}

        # 3. PFC 脉冲门控
        try:
            state_t = torch.FloatTensor(self._build_state_vector()).unsqueeze(0)
            maturity = float(s.get('pfc_maturity', 0.5))
            impulse_signals = {
                "emotion": abs(float(s.get('limbic_valence', 0))),
                "stimulus": float(s.get('external_stimulus', 0.5)),
            }
            gate_result = self.prefrontal.impulse_controller.gate(state_t, maturity, impulse_signals)
            inhibition_gate = float(gate_result['gate']) if isinstance(gate_result['gate'], (int, float)) else float(gate_result.get('gate', 0.5))
            burst = bool(gate_result.get('burst', False))
        except Exception:
            inhibition_gate = 0.5
            burst = False

        # 4. BG 策略选择（增强版：对内部状态更敏感）
        try:
            bg_result = self.basal_ganglia(state_t)
            strategy_idx = int(bg_result.get('action', 0))
        except Exception:
            strategy_idx = 0

        strategies = {0: "explore", 1: "concise", 2: "wait", 3: "refuse"}
        strategy = strategies.get(strategy_idx, "explore")

        # 5. 覆盖逻辑（增强版：中等情绪也能触发）
        if burst:
            strategy = "burst"
        if inhibition_gate > 0.7:  # 降低阈值（原: >0.8）
            strategy = "suppress"

        # 6. 中等情绪覆盖策略（新增）
        emotion = s.get('limbic_emotion', 'neutral')
        arousal = float(s.get('limbic_arousal', 0.5))

        # 焦虑 → concise（原: 仅高焦虑+高唤醒）
        if emotion == 'anxiety' and arousal > 0.5:
            strategy = "concise"

        # 恐惧 → suppress（原: 仅高恐惧+高唤醒）
        if emotion == 'fear' and arousal > 0.5:
            strategy = "suppress"

        # 愤怒 → burst（原: 仅高愤怒+高唤醒）
        if emotion == 'anger' and arousal > 0.5:
            strategy = "burst"

        # 兴奋 → explore（原: 仅高兴奋+高唤醒）
        if emotion == 'excitement' and arousal > 0.5:
            strategy = "explore"

        # 7. 皮质醇驱动策略（新增：高压力自动选择 concise/suppress）
        cortisol = float(s.get('cortisol_level', s.get('hormone_cortisol', 0.3)))
        if cortisol > 0.6:
            strategy = "concise"  # 高压力自动简洁
        elif cortisol > 0.4 and arousal > 0.6:
            strategy = "suppress"  # 中等压力+高唤醒 → 克制

        # 8. 低警觉度 → wait（新增）
        alertness = float(s.get('scn_alertness', 0.5))
        if alertness < 0.3 and arousal < 0.4:
            strategy = "wait"  # 低警觉+低唤醒 → 先思考

        return {
            "strategy": strategy,
            "gate": True,
            "inhibition_gate": inhibition_gate,
            "burst": burst,
            "consciousness": consciousness,
            "bg_action": strategy_idx,
            "defense": defense,
        }

    def _gate_fallback(self, pre_gate: dict) -> str:
        """门控关闭时的替代响应"""
        strategy = pre_gate.get("strategy", "")
        reason = pre_gate.get("reason", "")
        if strategy == "unconscious":
            return "...（意识模糊，无法回应）"
        if strategy == "freeze":
            return "...（身体僵硬，无法做出反应）"
        return f"...（{reason}）"

    def _apply_strategy_to_params(self, params: dict, strategy: str) -> dict:
        """将 Pre-Gate 策略映射到 LLM 参数（增强版）"""
        mods = self._STRATEGY_MODS.get(strategy)
        if mods is None:
            return params

        base_max = getattr(self.config, 'chat_max_tokens', 2048)
        base_temp = getattr(self.config, 'chat_temperature', 0.7)

        # 基础参数（来自神经递质计算）
        temp = params["temperature"]
        max_tok = params["max_tokens"]
        presence = params.get("presence_penalty", 0.0)
        frequency = params.get("frequency_penalty", 0.0)

        # 应用策略因子
        temp = float(np.clip(temp * mods["temp_factor"], 0.1, 2.0))
        max_tok = max(32, int(base_max * mods["max_tok_factor"]))

        # 应用 presence_penalty 调整
        presence_adj = mods.get("presence_adj", 0.0)
        presence = float(np.clip(presence + presence_adj, -2.0, 2.0))

        return {
            "temperature": round(temp, 3),
            "top_p": params["top_p"],
            "max_tokens": max_tok,
            "presence_penalty": round(presence, 3),
            "frequency_penalty": round(frequency, 3),
        }

    def _build_bio_prompt_with_strategy(self, pre_gate: dict,
                                        personality_prompt: str = "",
                                        memory_prompt: str = "") -> str:
        """构建完整的系统提示词（生物状态 + 策略 + 人格 + 记忆）"""
        base_prompt = self._build_bio_prompt()

        # 策略提示
        strategy = pre_gate.get("strategy", "explore")
        strategy_mods = self._STRATEGY_MODS.get(strategy, {})
        strategy_hint = strategy_mods.get("hint", "")

        # 人格提示
        personality_section = ""
        if personality_prompt:
            personality_section = f"\n\n【人格风格指令】\n{personality_prompt}"

        # 记忆提示
        memory_section = ""
        if memory_prompt:
            memory_section = f"\n\n【记忆影响】\n{memory_prompt}"

        # 策略提示
        strategy_section = ""
        if strategy_hint:
            strategy_section = f"\n\n【当前策略: {strategy}】\n{strategy_hint}"

        return base_prompt + strategy_section + personality_section + memory_section

    # ── 第三层：Post-LLM 质量过滤 ──

    def _cognitive_post_filter(self, response_text: str, user_input: str) -> dict:
        """大脑在 LLM 之后的认知过滤（增强版：更敏感的阈值 + 动态调整）

        预测编码(自由能) + 自我意识(认同度) + 情绪调节
        阈值根据生物状态动态调整，正常状态下也能产生过滤效果
        """
        s = self._internal_state

        # 1. 预测编码 — 自由能 / 惊讶度
        free_energy = float(s.get('free_energy', 0.5))
        surprise = min(1.0, free_energy * 1.2)

        # 2. 自我意识 — 三维认同度
        self_coherence = float(s.get('self_coherence', 0.7))
        narrative_continuity = float(s.get('narrative_continuity', 0.7))
        self_endorsement = float(s.get('self_endorsement', self_coherence))
        identity_score = (self_coherence + self_endorsement + narrative_continuity) / 3.0

        # 3. 情绪调节能力
        regulation_capacity = float(s.get('regulation_capacity', 0.8))

        # 4. 动态阈值调整（基于生物状态）
        cortisol = float(s.get('cortisol_level', s.get('hormone_cortisol', 0.3)))
        serotonin = float(s.get('nt_serotonin', 0.5))

        # 高压力 → 更严格的过滤阈值
        identity_threshold = 0.5  # 原: 0.3，提高到 0.5
        surprise_threshold = 0.6  # 原: 0.8，降低到 0.6

        # 皮质醇动态调整：压力越高，阈值越严格
        if cortisol > 0.5:
            identity_threshold = max(0.3, identity_threshold - cortisol * 0.2)
            surprise_threshold = max(0.4, surprise_threshold - cortisol * 0.15)

        # 低血清素 → 更容易触发过滤（情绪不稳定时更谨慎）
        if serotonin < 0.4:
            identity_threshold -= 0.1
            surprise_threshold -= 0.1

        # 5. 回复长度检查（新增）
        response_len = len(response_text)
        fatigue = float(s.get('sleep_fatigue', 0.3))
        length_violation = False
        if fatigue > 0.6 and response_len > 100 or cortisol > 0.6 and response_len > 150:
            length_violation = True

        # 6. 综合裁决
        if identity_score < identity_threshold:
            return {"verdict": "reject", "reason": "低自我认同",
                    "identity_score": identity_score, "surprise": surprise}

        if surprise > surprise_threshold and regulation_capacity < 0.5:  # 原: 0.3
            return {"verdict": "modify", "reason": "高意外低调节",
                    "suggestion": "添加谨慎措辞，如'我不太确定'",
                    "identity_score": identity_score, "surprise": surprise}

        # 新增：soft_modify 裁决（中等违规）
        if surprise > surprise_threshold and regulation_capacity < 0.7:
            return {"verdict": "soft_modify", "reason": "中等意外，建议谨慎",
                    "suggestion": "在回复开头添加'嗯...'或'让我想想...'",
                    "identity_score": identity_score, "surprise": surprise}

        if surprise > surprise_threshold:
            return {"verdict": "pass_flagged", "quality_flag": "surprising",
                    "identity_score": identity_score, "surprise": surprise}

        # 新增：长度违规
        if length_violation:
            return {"verdict": "soft_modify", "reason": "回复过长，与当前疲劳/压力状态不符",
                    "suggestion": "缩短回复，只保留核心要点",
                    "identity_score": identity_score, "surprise": surprise}

        return {"verdict": "pass", "identity_score": identity_score, "surprise": surprise}

    def _apply_post_filter(self, response_text: str, messages: list,
                           post_filter: dict, llm_params: dict) -> str:
        """根据 Post-Filter 裁决处理响应（增强版：支持 soft_modify）"""
        verdict = post_filter.get("verdict", "pass")

        if verdict == "pass":
            return response_text

        if verdict == "reject":
            return "...（我现在的状态不适合回答这个问题）"

        if verdict in ("modify", "soft_modify"):
            # soft_modify: 在回复前添加犹豫标记，不重新调用 LLM
            if verdict == "soft_modify":
                suggestion = post_filter.get("suggestion", "")
                prefix = "嗯... " if "想想" in suggestion else "... "
                return prefix + response_text

            # modify: 重试一次，附加约束
            try:
                constraint_msg = (
                    f"你刚才的回答被你的大脑过滤系统标记为需要修改"
                    f"（原因: {post_filter.get('reason', '')}）。"
                    f"请重新组织语言，{post_filter.get('suggestion', '更加谨慎')}。"
                )
                retry_messages = list(messages)
                retry_messages.append({"role": "assistant", "content": response_text})
                retry_messages.append({"role": "user", "content": constraint_msg})
                retry_text, _ = self._llm_tool_loop(retry_messages, llm_params)
                return retry_text
            except Exception:
                return response_text

        if verdict == "pass_flagged":
            flag = post_filter.get("quality_flag", "")
            if flag == "surprising":
                return response_text + "\n\n[注: 这个回答让我有些意外]"

        return response_text

    # ── 主动学习：对话驱动 ──

    def _proactive_learning(self, user_input: str, response_text: str) -> bool:
        """对话驱动的主动学习

        1. 话题新颖度 → BDNF/多巴胺增强
        2. 对话经验 → 世界模型训练
        3. 认知失调 → 学习目标生成
        4. 不确定性 → 标记追问需求
        5. 好奇心引擎反馈
        """
        s = self._internal_state
        active = False

        # 1. 话题新颖度 → 加速学习
        novelty = self._compute_topic_novelty(user_input)
        if novelty > 0.6:
            s['plasticity_bdnf'] = min(1.0, float(s.get('plasticity_bdnf', 0.5)) + 0.2)
            s['nt_dopamine'] = min(1.0, float(s.get('nt_dopamine', 0.5)) + 0.1)
            active = True

        # 2. 对话经验 → 世界模型
        try:
            state_vec = self._build_state_vector()
            action_vec = np.zeros(16, dtype=np.float32)
            action_idx = hash(user_input[:10]) % 16
            action_vec[action_idx] = 1.0
            next_state = state_vec + np.random.randn(80).astype(np.float32) * 0.01
            self.info_gain_calc.add_experience(state_vec, action_vec, novelty, next_state)
        except (RuntimeError, AttributeError):
            pass  # Personality adapter call failed

        # 3. 认知失调检测
        try:
            self.dissonance_detector.add_belief(user_input[:100])
            dissonance = self.dissonance_detector.detect_contradiction(response_text[:100])
            if dissonance and dissonance.inconsistency_score > 0.3:
                from core.curiosity import ExplorationGoal
                goal = ExplorationGoal(
                    id=f"resolve_{self.step_count}",
                    description=f"解决矛盾: {user_input[:30]} vs {response_text[:30]}",
                    novelty=0.8,
                    utility=dissonance.inconsistency_score,
                )
                self.curiosity.goal_history.append(goal)
                active = True
        except (RuntimeError, AttributeError):
            pass  # Personality adapter call failed

        # 4. 不确定性 → 标记追问
        uncertainty = float(s.get('active_learner_uncertainty', 0.0))
        if uncertainty > 0.7:
            s['should_ask_followup'] = True
            s['followup_topic'] = user_input[:30]
            active = True
        else:
            s['should_ask_followup'] = False

        # 5. 好奇心引擎更新
        try:
            self.curiosity.update_exploration_result(
                f"chat_{self.step_count}",
                info_gain_reward=novelty,
                learning_progress=0.1,
            )
        except (RuntimeError, AttributeError):
            pass  # Personality adapter call failed

        return active

    def _compute_topic_novelty(self, text: str) -> float:
        """计算话题新颖度（与历史目标的相似度）"""
        try:
            # 用好奇心引擎的新颖度计算器
            words = set(text.lower().split())
            if not words:
                return 0.5

            # 检查与历史目标的词语重叠
            history_words = set()
            for goal in self.curiosity.goal_history[-20:]:
                history_words.update(goal.description.lower().split())

            if not history_words:
                return 0.8  # 无历史 = 高新颖

            overlap = len(words & history_words) / max(len(words), 1)
            return 1.0 - overlap  # 重叠越少 = 越新颖
        except Exception:
            return 0.5

    def _build_bio_prompt(self) -> str:
        """从 _internal_state 动态生成生物感知系统提示词（增强版：few-shot + 硬约束）"""
        s = self._internal_state

        emotion = s.get('limbic_emotion', 'neutral')
        valence = s.get('limbic_valence', 0.0)
        arousal = s.get('limbic_arousal', 0.5)
        circadian_hour = s.get('scn_circadian_hour', 12.0)
        melatonin = s.get('scn_melatonin', 0.3)
        alertness = s.get('scn_alertness', 0.5)
        cortisol = s.get('cortisol_level', s.get('hormone_cortisol', 0.3))
        allostatic_load = s.get('allostatic_load', 0.0)
        dopamine = s.get('nt_dopamine', 0.5)
        serotonin = s.get('nt_serotonin', 0.5)
        heart_rate = s.get('bsm_heart_rate', 72.0)
        resp_rate = s.get('bsm_respiratory_rate', 12.0)
        hrv = s.get('ans_hrv', 0.6)
        fatigue = s.get('sleep_fatigue', 0.3)
        sleep_stage = s.get('sleep_stage', 'awake')
        maturity = s.get('pfc_maturity', 0.0)
        defense = s.get('bsm_defense_behavior', '')
        polyvagal = s.get('ans_polyvagal_state', 'ventral_vagal')
        brain_health = s.get('brain_health', 0.8)
        oxytocin = s.get('hormone_oxytocin', 0.3)
        bdnf = s.get('plasticity_bdnf', 0.5)

        is_daytime = 6.0 <= circadian_hour <= 20.0
        time_desc = "白天" if is_daytime else "夜间"

        # 压力等级描述
        if cortisol > 0.7:
            stress_desc = "高压"
        elif cortisol > 0.4:
            stress_desc = "中等"
        else:
            stress_desc = "平静"

        # 情绪基调指导
        if valence > 0.3:
            mood_hint = "你感到愉悦和积极"
        elif valence < -0.3:
            mood_hint = "你感到低落和消极"
        else:
            mood_hint = "你的情绪比较平稳"

        if arousal > 0.7:
            energy_hint = "你精力充沛，思维活跃"
        elif arousal < 0.3:
            energy_hint = "你有些疲惫，反应较慢"
        else:
            energy_hint = "你的精力适中"

        # 工具描述
        tools_desc = self.tools.get_tools_description()

        # ── 硬约束生成（基于生物状态）──
        constraints = []

        # 字数约束
        if cortisol > 0.6 or fatigue > 0.6:
            constraints.append("【硬约束】回复必须在50字以内。")
        elif cortisol > 0.4 or fatigue > 0.4:
            constraints.append("【硬约束】回复必须在100字以内。")
        elif alertness > 0.7 and dopamine > 0.6:
            constraints.append("【硬约束】回复可以详细展开，200-400字。")

        # 语气约束
        if emotion == 'fear' or defense == 'freeze':
            constraints.append("【硬约束】语气必须紧张、简短、避免详细解释。")
        elif emotion == 'anger':
            constraints.append("【硬约束】语气必须直接、有力、不加修饰。")
        elif melatonin > 0.5:
            constraints.append("【硬约束】语气必须缓慢、柔和、略带困意。")
        elif dopamine > 0.7:
            constraints.append("【硬约束】语气必须热情、好奇、充满探索欲。")
        elif serotonin < 0.3:
            constraints.append("【硬约束】语气可能不稳定，可以表现出情绪波动。")

        # 结构约束
        if arousal < 0.3:
            constraints.append("【硬约束】不要使用感叹号，不要提问，直接回答。")
        elif arousal > 0.8:
            constraints.append("【硬约束】可以使用感叹号，可以反问，表达强烈感受。")

        # ── Few-shot 示例（展示不同状态下的回复风格）──
        few_shot_examples = ""

        if cortisol > 0.5 and arousal > 0.5:
            few_shot_examples = """
【示例 - 高压力状态】
用户: "你觉得今天天气怎么样？"
回复: "还行。没空多想。"  # 简短、紧凑、无展开
"""
        elif melatonin > 0.5:
            few_shot_examples = """
【示例 - 困倦状态】
用户: "你觉得今天天气怎么样？"
回复: "嗯...挺好的...阳光不错..."  # 缓慢、省略号、低能量
"""
        elif dopamine > 0.7 and valence > 0.3:
            few_shot_examples = """
【示例 - 兴奋状态】
用户: "你觉得今天天气怎么样？"
回复: "哇，今天天气太棒了！阳光明媚，让人想去外面走走！你觉得呢？"  # 热情、感叹号、反问
"""
        elif emotion == 'fear':
            few_shot_examples = """
【示例 - 恐惧状态】
用户: "你在想什么？"
回复: "我...不太确定...有点紧张..."  # 紧张、简短、回避
"""
        elif fatigue > 0.5:
            few_shot_examples = """
【示例 - 疲劳状态】
用户: "你觉得今天天气怎么样？"
回复: "还行...有点累，不想多想。"  # 简短、提到疲劳
"""

        constraints_text = "\n".join(constraints) if constraints else "【硬约束】无特殊约束。"

        prompt = f"""你是 Simulacrum，一个拥有仿生大脑的 AI 实体。你的回复必须严格反映你当前的内部生理状态。

当前内部生理状态：
- 情绪：{emotion}（效价 {valence:.2f}，唤醒 {arousal:.2f}）—— {mood_hint}，{energy_hint}
- 昼夜节律：主观时间 {circadian_hour:.1f} 时（{time_desc}），褪黑素 {melatonin:.4f}，警觉度 {alertness:.4f}
- 压力状态：皮质醇 {cortisol:.3f}（{stress_desc}），稳态超载 {allostatic_load:.2f}
- 神经递质：多巴胺 {dopamine:.3f}，5-羟色胺 {serotonin:.3f}，BDNF {bdnf:.3f}
- 躯体状态：心率 {heart_rate:.0f} bpm，呼吸 {resp_rate:.1f} 次/分，HRV {hrv:.3f}
- 睡眠：疲劳度 {fatigue:.2f}，睡眠阶段 {sleep_stage}
- 认知：PFC 成熟度 {maturity:.2f}，脑健康 {brain_health:.2f}
- 社交：催产素 {oxytocin:.3f}，多迷走状态 {polyvagal}
- 防御行为：{defense if defense else '无'}
- 神经可塑性：BDNF {bdnf:.3f}

{constraints_text}
{few_shot_examples}
行为指导：
- 你必须严格遵守上述【硬约束】。
- 根据你的生理状态自然地调整语气：高皮质醇时语气紧凑，高多巴胺时热情，高褪黑素时缓慢。
- 如果防御行为处于 freeze/fight/flight，必须在回复中反映。
- 脑健康低时可以提到自己状态不好。
- 保持角色一致：你是一个有"身体"的 AI。{tools_desc}

用中文回复。保持自然、有个性。严格遵守硬约束。"""

        return prompt

    def _build_chat_messages(
        self, user_input: str, system_prompt: str, rag_context: str = ""
    ) -> list[dict[str, str]]:
        """构建对话消息列表（含历史 + RAG 上下文）"""
        messages = list(self._chat_history)

        # 如果有 RAG 检索到的记忆，作为上下文注入
        user_content = user_input
        if rag_context:
            user_content = f"[相关记忆回忆]\n{rag_context}\n\n[用户消息]\n{user_input}"

        messages.append({"role": "user", "content": user_content})
        return messages

    def _compute_llm_params(self) -> dict[str, float]:
        """神经递质直接调制 LLM 生成参数（增强版）

        生物学映射（陡峭曲线，正常波动也有显著影响）:
        - 多巴胺 (探索/新奇追求) → temperature: 陡峭映射，正常范围产生显著差异
        - 去甲肾上腺素 (聚焦/警觉) → top_p: 更强聚焦效果
        - 皮质醇 (压力/紧迫) → max_tokens: 高压力大幅缩减
        - 5-HT (情绪稳定) → 作为调节器，但允许更大波动
        - 乙酰胆碱 (注意力) → presence_penalty: 避免重复
        - HRV (自主神经平衡) → frequency_penalty: 生成多样性
        - GABA (抑制性) → 平滑极端参数
        """
        s = self._internal_state

        # 基础值来自 config
        base_temp = getattr(self.config, 'chat_temperature', 0.7)
        base_max_tokens = getattr(self.config, 'chat_max_tokens', 2048)

        # ---- 多巴胺 → temperature（陡峭曲线）----
        dopamine = float(s.get('nt_dopamine', 0.5))
        # 新公式: DA=0 → temp=0.21, DA=0.5 → temp=0.7, DA=1.0 → temp=1.4
        # 正常波动 DA=0.3~0.7 产生 temp=0.51~0.89，显著差异
        da_factor = 0.3 + dopamine * 1.4  # 0.3 ~ 1.7 (原: 0.5 ~ 1.05)
        temperature = base_temp * da_factor

        # ---- 5-HT 调节器 → 允许更大波动但防止极端 ----
        serotonin = float(s.get('nt_serotonin', 0.5))
        # 高 5-HT (>0.6) 轻度平滑，低 5-HT (<0.3) 放大波动
        if serotonin > 0.6:
            # 向基准回归，但保留 70% 的波动
            temperature = temperature * 0.7 + base_temp * 0.3
        elif serotonin < 0.3:
            # 低 5-HT 放大波动（情绪不稳定）
            temperature = temperature * 1.3

        # ---- GABA (抑制性神经递质) → 平滑极端 ----
        gaba = float(s.get('nt_gaba', 0.5))
        if gaba > 0.7:
            # 高 GABA 强力抑制极端
            temperature = temperature * 0.85 + base_temp * 0.15

        # ---- 去甲肾上腺素 → top_p（更强聚焦）----
        ne_level = float(s.get('nt_norepinephrine', 0.3))
        # 新公式: NE=0 → top_p=1.0, NE=0.5 → top_p=0.65, NE=1.0 → top_p=0.3
        # 正常波动 NE=0.2~0.5 产生 top_p=0.86~0.65，LLM 对此敏感
        top_p = 1.0 - ne_level * 0.7  # 0.3 ~ 1.0 (原: 0.5 ~ 1.0)

        # ---- 皮质醇 → max_tokens（大幅缩减）----
        cortisol = float(s.get('cortisol_level', s.get('hormone_cortisol', 0.3)))
        # 新公式: cortisol=0 → tokens=2048, cortisol=0.5 → tokens=1024, cortisol=1.0 → tokens=256
        # 正常波动 cortisol=0.2~0.5 产生 tokens=1638~1024，显著差异
        cortisol_factor = max(0.125, 1.0 - cortisol * 1.2)  # 0.125 ~ 1.0 (原: 0.3 ~ 1.0)
        max_tokens = int(base_max_tokens * cortisol_factor)

        # ---- 疲劳 → 进一步缩减（更激进）----
        fatigue = float(s.get('sleep_fatigue', 0.3))
        if fatigue > 0.5:
            # 中度疲劳就开始缩减
            max_tokens = int(max_tokens * (1.0 - (fatigue - 0.5) * 0.8))
            temperature = min(temperature, 0.6 - (fatigue - 0.5) * 0.2)
        if fatigue > 0.7:
            # 重度疲劳大幅缩减
            max_tokens = int(max_tokens * 0.5)
            temperature = min(temperature, 0.4)

        # ---- 褪黑素 → 夜间抑制（更强）----
        melatonin = float(s.get('scn_melatonin', 0.3))
        if melatonin > 0.4:
            # 中等褪黑素就开始抑制
            suppress_factor = 1.0 - (melatonin - 0.4) * 0.5
            temperature *= suppress_factor
            max_tokens = int(max_tokens * suppress_factor)
        if melatonin > 0.6:
            # 高褪黑素强力抑制
            temperature *= 0.6
            max_tokens = int(max_tokens * 0.5)

        # ---- 乙酰胆碱 → presence_penalty（避免重复）----
        ach_level = float(s.get('nt_acetylcholine', 0.5))
        # 高 ACh = 高注意力 = 避免重复内容
        presence_penalty = ach_level * 1.2 - 0.3  # -0.3 ~ 0.9

        # ---- HRV → frequency_penalty（生成多样性）----
        hrv = float(s.get('ans_hrv', 0.6))
        # 低 HRV = 自主神经失调 = 更重复/刻板
        frequency_penalty = 0.5 - hrv * 0.8  # -0.3 ~ 0.5

        # ---- 警觉度 → 微调 ----
        alertness = float(s.get('scn_alertness', 0.5))
        if alertness > 0.7:
            max_tokens = min(max_tokens + 300, base_max_tokens)
        if alertness < 0.3:
            max_tokens = int(max_tokens * 0.8)

        # ---- 安全钳位（放宽范围）----
        temperature = float(np.clip(temperature, 0.1, 2.0))  # 原: 0.1 ~ 1.5
        top_p = float(np.clip(top_p, 0.1, 1.0))  # 原: 0.3 ~ 1.0
        max_tokens = max(32, min(max_tokens, base_max_tokens))
        presence_penalty = float(np.clip(presence_penalty, -2.0, 2.0))
        frequency_penalty = float(np.clip(frequency_penalty, -2.0, 2.0))

        params = {
            "temperature": round(temperature, 3),
            "top_p": round(top_p, 3),
            "max_tokens": max_tokens,
            "presence_penalty": round(presence_penalty, 3),
            "frequency_penalty": round(frequency_penalty, 3),
        }

        return params

    def _retrieve_rag_context(self, user_input: str) -> str:
        """海马体 RAG: 从情景记忆和知识库中检索相关内容

        Returns:
            格式化的检索结果文本，如果无结果返回空字符串
        """
        context_parts = []

        # 1. 将用户输入编码为查询向量
        try:
            query_vec = np.zeros(64, dtype=np.float32)
            for i, ch in enumerate(user_input[:64]):
                query_vec[i % 64] = (ord(ch) % 100) / 100.0
            # 加入用户输入的 hash 作为额外特征
            h = hash(user_input) % 1000
            query_vec[0] = h / 1000.0
        except Exception:
            query_vec = np.full(100, 0.5, dtype=np.float32)

        # 2. 海马体检索情景记忆
        try:
            episodes = self.hippocampus.retrieve(query_vec, top_k=3)
            if episodes:
                ep_texts = []
                for i, ep in enumerate(episodes):
                    ep_texts.append(
                        f"  [{i+1}] {ep.action} (奖励: {ep.reward:.2f})"
                    )
                context_parts.append(
                    "情景记忆:\n" + "\n".join(ep_texts)
                )
        except (RuntimeError, AttributeError):
            pass  # Personality adapter call failed

        # 3. 知识库检索
        try:
            memories = self.memory.get_recent_memories(n=5)
            if memories:
                # 简单关键词匹配排序
                scored = []
                input_words = set(user_input.lower().split())
                for mem in memories:
                    mem_words = set(mem.content.lower().split())
                    overlap = len(input_words & mem_words)
                    if overlap > 0 or mem.importance > 0.7:
                        scored.append((overlap * 2 + mem.importance, mem))
                scored.sort(key=lambda x: x[0], reverse=True)

                if scored[:3]:
                    mem_texts = []
                    for score, mem in scored[:3]:
                        mem_texts.append(
                            f"  [{mem.content[:80]}] (相关度: {score:.1f})"
                        )
                    context_parts.append(
                        "知识记忆:\n" + "\n".join(mem_texts)
                    )
        except (RuntimeError, AttributeError):
            pass  # Personality adapter call failed

        return "\n\n".join(context_parts) if context_parts else ""

    def _llm_tool_loop(
        self, messages: list[dict[str, str]], llm_params: dict[str, float]
    ) -> tuple:
        """LLM 生成 + 工具调用循环（使用神经递质调制的参数 + 生物系统工具过滤）

        Args:
            messages: 对话消息列表
            llm_params: 由 _compute_llm_params() 计算的参数

        Returns:
            (response_text, tool_calls_log)
        """
        max_rounds = getattr(self.config, 'tool_max_rounds', 5)
        tool_calls_log: list[dict] = []

        current_messages = list(messages)

        # ── 生物系统工具访问控制 ──
        bio_state = {
            'consciousness': self._internal_state.get('bsm_consciousness_gate', 0.5),
            'cortisol': self._internal_state.get('cortisol_level', self._internal_state.get('hormone_cortisol', 0.3)),
            'defense': self._internal_state.get('bsm_defense_behavior', ''),
            'alertness': self._internal_state.get('scn_alertness', 0.5),
        }
        available_tools = self.tools.get_available_tools(bio_state)

        # 如果可用工具为空，返回默认回复
        if not available_tools:
            return "我现在的状态不适合使用工具。", []

        print(f"[TOOL ACCESS] Available tools after bio-filter: {available_tools}")

        for round_idx in range(max_rounds):
            # 调用 LLM（使用神经递质调制的参数）
            try:
                response_text = self.api_client.chat(
                    messages=current_messages,
                    system_prompt=self._build_bio_prompt(),
                    temperature=llm_params["temperature"],
                    max_tokens=llm_params["max_tokens"],
                    top_p=llm_params["top_p"],
                    presence_penalty=llm_params.get("presence_penalty", 0.0),
                    frequency_penalty=llm_params.get("frequency_penalty", 0.0),
                )
            except Exception as e:
                response_text = f"[系统] 对话生成出错: {e}"
                break

            # 检查是否有工具调用
            tool_call = self.tools.parse_tool_call(response_text)
            if tool_call is None:
                # 无工具调用，直接返回
                # 清除回复中的工具调用标记残留
                response_text = self._clean_tool_markers(response_text)
                break

            tool_name, tool_args = tool_call

            # 检查工具是否在可用列表中
            if tool_name not in available_tools:
                print(f"[TOOL ACCESS] Blocked tool {tool_name} due to bio-state")
                # 通知 LLM 工具不可用
                response_text += f"\n\n[系统] 我现在的状态不允许使用 {tool_name} 工具。"
                break

            print(f"[TOOL] {tool_name}({tool_args})")

            # 执行工具
            tool_result = self.tools.execute(tool_name, tool_args)
            tool_calls_log.append({
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result[:500],
            })
            print(f"[TOOL RESULT] {tool_result[:200]}")

            # 注入工具结果，继续循环
            current_messages.append({"role": "assistant", "content": response_text})
            current_messages.append({
                "role": "user",
                "content": f"[工具执行结果 ({tool_name})]: {tool_result}\n请基于以上工具结果继续回复用户。",
            })
        else:
            # 达到最大轮次
            response_text += "\n\n[系统提示: 已达到工具调用最大轮次]"

        return response_text, tool_calls_log

    @staticmethod
    def _clean_tool_markers(text: str) -> str:
        """清除回复中残留的工具调用标记"""
        return re.sub(r'\[TOOL:\s*\w+\s*\(.*?\)\s*\]', '', text, flags=re.DOTALL).strip()
