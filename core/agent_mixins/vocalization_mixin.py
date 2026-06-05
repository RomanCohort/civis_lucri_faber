"""VocalizationMixin — 发音系统辅助方法

从 monolithic agent.py 中提取的发音相关方法:
- _prepare_vocalization_input()
- _get_emotion_vector()
- _get_censor_result_dict()
"""

import torch


class VocalizationMixin:
    """发音系统 Mixin — 音素准备、情绪向量提取、Censor结果提取

    使用 self 访问 Simulacrum 实例的属性:
    - self._internal_state: 内部状态字典
    - self.censor: Censor微表情感知模块
    """

    def _prepare_vocalization_input(
        self, user_input: str = None, response_text: str = None
    ) -> torch.Tensor | None:
        """从文本输入准备音素索引张量

        优先使用 response_text（LLM输出），回退到 user_input。
        返回 [1, T] 的音素索引张量，如果无法生成则返回 None。
        """
        source_text = response_text or user_input
        if not source_text:
            return None

        # 简单文本→音素（生产环境应使用CMU dict / g2p）
        from core.vocalization import text_to_phoneme_indices
        indices = text_to_phoneme_indices(source_text)
        if not indices:
            return None

        # 截断到合理长度（避免过长序列）
        max_phonemes = 60
        indices = indices[:max_phonemes]
        return torch.tensor([indices], dtype=torch.long)

    def _get_emotion_vector(self) -> torch.Tensor:
        """从内部状态提取8维情绪向量 (Plutchik)

        8维: joy, sadness, anger, fear, trust, disgust, surprise, anticipation
        """
        s = self._internal_state
        emotion = s.get('limbic_emotion', 'neutral')
        valence = float(s.get('limbic_valence', 0.0))
        arousal = float(s.get('limbic_arousal', 0.5))

        vec = torch.zeros(1, 8)
        emotion_map = {
            'joy': 0, 'happiness': 0,
            'sadness': 1,
            'anger': 2,
            'fear': 3,
            'trust': 4,
            'disgust': 5,
            'surprise': 6,
            'anticipation': 7,
        }
        idx = emotion_map.get(emotion, 0)
        if emotion in emotion_map:
            vec[0, idx] = abs(valence) * arousal
        else:
            # neutral: distributed activation
            vec[0, :] = 0.1
        return vec

    def _get_censor_result_dict(self) -> dict | None:
        """从 Censor 模块提取结果字典，供 ChatResponse 返回"""
        result = self.censor.get_last_result()
        if result is None:
            return None
        return {
            "me_predicted": result.me_predicted,
            "me_confidence": result.me_confidence,
            "me_category": self.censor._get_me_category(result.me_predicted),
            "au_active": result.au_active,
            "au_dominant": result.au_dominant,
            "au_dominant_intensity": result.au_dominant_intensity,
            "apex_frame": result.apex_frame,
            "dominant_expert": result.dominant_expert,
            "emotion_map": self.censor._compute_emotion_map(result.au_intensities),
            "template_report": result.template_report,
            "inference_time_ms": result.inference_time_ms,
        }