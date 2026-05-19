"""LLM临床评估器 — 用LLM对治疗过程进行结构化临床评估。

将PsychometricSnapshot + 神经递质状态 + 药物/疗法指标
组装为临床评估提示词，调用LLM获取结构化JSON评估。

支持两种模式:
  - API模式: 通过APIClient调用真实LLM
  - Mock模式: 基于规则生成模拟评估（无需API key）

参考文献:
  - Busner & Targum (2006) CGI: Innov Clin Neurosci 3(1):28-32
  - Guy (1976) ECDEU Assessment Manual: CGI-I
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from simulacrum.core.psychometric_indicators import PsychometricSnapshot


@dataclass
class LLMEvaluationResult:
    """LLM临床评估结果。"""
    timepoint: int
    time_h: float

    # 结构化评估字段
    diagnostic_summary: str           # 诊断摘要
    severity_assessment: str          # minimal/mild/moderate/moderately_severe/severe
    treatment_progress: str           # no_change/minimal/moderate/significant/full_remission
    relapse_risk: str                 # low/moderate/high
    side_effect_risk: str             # none/low/moderate/high
    recommended_adjustments: List[str] = field(default_factory=list)
    clinical_global_impression: float = 4.0   # CGI-I [1,7], 1=very much improved
    functional_improvement: float = 0.0       # [0,1]

    # 元数据
    raw_response: str = ""
    evaluation_mode: str = "mock"     # "api" / "mock"
    error: Optional[str] = None


# ── 临床评估提示词模板 ──

EVAL_PROMPT_TEMPLATE = """你是临床精神药理学专家，正在评估一个计算模拟的精神科治疗过程。

## 患者档案
- 诊断: {condition} ({severity})
- 治疗方案: {drug_info}
- 心理治疗: {therapy_info}

## 当前时间点
- 模拟时间: {time_h:.1f} 小时 (第 {step} 步)

## 心理测量指标
- 抑郁严重度 (PHQ-9模拟): {phq9:.1f} / 10
- 焦虑水平 (GAD-7模拟): {gad7:.1f} / 10
- 认知功能 (MoCA模拟): {cognitive:.1f} / 10
- 情绪调节困难 (DERS模拟): {emotion_reg:.1f} / 10
- 社会功能: {social:.1f} / 10
- 综合症状严重度: {global_severity:.2f} / 1.0

## 神经递质状态
- 多巴胺 (DA): {da:.3f}
- 5-羟色胺 (5-HT): {sht:.3f}
- 去甲肾上腺素 (NE): {ne:.3f}
- GABA: {gaba:.3f}
- 皮质醇: {cort:.3f}

## 脑区功能
- 前额叶成熟度: {pfc:.3f}
- 边缘系统效价: {valence:.3f}
- 边缘系统唤醒: {arousal:.3f}
- HPA轴反应性: {hpa:.3f}

## 药物/治疗状态
- 药物浓度: {conc:.4f} mg/L
- PD效应强度: {pd_effect:.4f}
- 治疗技能水平: {therapy_skill:.3f}
- 协同因子: {synergy:.3f}

## 基线对比
- PHQ-9基线: {phq9_baseline:.1f} → 当前: {phq9:.1f} (变化: {phq9_delta:+.1f})
- GAD-7基线: {gad7_baseline:.1f} → 当前: {gad7:.1f} (变化: {gad7_delta:+.1f})

请以JSON格式输出临床评估，严格遵循以下结构:
```json
{{
  "diagnostic_summary": "简要诊断描述(1-2句)",
  "severity_assessment": "minimal/mild/moderate/moderately_severe/severe",
  "treatment_progress": "no_change/minimal/moderate/significant/full_remission",
  "relapse_risk": "low/moderate/high",
  "side_effect_risk": "none/low/moderate/high",
  "recommended_adjustments": ["建议1", "建议2"],
  "clinical_global_impression": 4,
  "functional_improvement": 0.0
}}
```

注意:
- clinical_global_impression 范围 [1,7]: 1=显著改善, 2=很大改善, 3=轻微改善, 4=无变化, 5=轻微恶化, 6=很大恶化, 7=严重恶化
- functional_improvement 范围 [0,1]: 0=无改善, 1=完全恢复功能
- 评估应基于指标变化趋势，而非绝对值
- 考虑药物浓度是否在治疗窗内
- 考虑协同因子是否表明联合治疗有效"""


class LLMEvaluator:
    """LLM临床评估器。

    用法:
        evaluator = LLMEvaluator(api_client=client)
        result = evaluator.evaluate(indicators=snap, condition="MDD", ...)
    """

    def __init__(
        self,
        api_client=None,
        mock_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        """
        Args:
            api_client: APIClient实例 (来自utils.api_client)
            mock_mode: True=不调用API，用规则生成评估
            temperature: LLM采样温度 (低=更确定性)
            max_tokens: 最大生成token数
        """
        self.api_client = api_client
        self.mock_mode = mock_mode or (api_client is None)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def evaluate(
        self,
        indicators: PsychometricSnapshot,
        condition: str,
        severity: str,
        drug_info: str = "",
        therapy_info: str = "",
        baseline: Optional[Dict[str, float]] = None,
        nt_state: Optional[Dict[str, float]] = None,
    ) -> LLMEvaluationResult:
        """评估单个时间点的治疗状态。

        Args:
            indicators: 心理测量指标快照
            condition: 诊断 (MDD/GAD/PTSD/BPD/OCD)
            severity: 严重度 (minimal/mild/moderate/severe)
            drug_info: 药物信息字符串
            therapy_info: 疗法信息字符串
            baseline: 基线指标 {"phq9_baseline": ..., "gad7_baseline": ...}
            nt_state: 神经递质状态 {"nt_dopamine": ..., ...}

        Returns:
            LLMEvaluationResult
        """
        if self.mock_mode:
            return self._mock_evaluate(
                indicators, condition, severity, baseline
            )

        # 构建提示词
        prompt = self._build_prompt(
            indicators, condition, severity,
            drug_info, therapy_info, baseline, nt_state,
        )

        # 调用API
        try:
            response = self.api_client.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="你是临床精神药理学专家，擅长评估计算模拟的精神科治疗。请严格按JSON格式输出。",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            return LLMEvaluationResult(
                timepoint=indicators.step,
                time_h=indicators.time_h,
                diagnostic_summary=f"API调用失败: {e}",
                severity_assessment="unknown",
                treatment_progress="no_change",
                relapse_risk="high",
                side_effect_risk="unknown",
                evaluation_mode="api",
                error=str(e),
            )

        # 解析JSON响应
        return self._parse_response(response, indicators)

    def evaluate_trajectory(
        self,
        trajectory: List[PsychometricSnapshot],
        condition: str,
        severity: str,
        drug_info: str = "",
        therapy_info: str = "",
        baseline: Optional[Dict[str, float]] = None,
        nt_trajectory: Optional[List[Dict[str, float]]] = None,
    ) -> List[LLMEvaluationResult]:
        """评估多个时间点的治疗轨迹。

        Args:
            trajectory: 心理测量指标快照列表
            condition: 诊断
            severity: 严重度
            drug_info: 药物信息
            therapy_info: 疗法信息
            baseline: 基线指标
            nt_trajectory: 每个时间点的神经递质状态列表

        Returns:
            评估结果列表
        """
        results: List[LLMEvaluationResult] = []

        for i, snap in enumerate(trajectory):
            nt_state = None
            if nt_trajectory and i < len(nt_trajectory):
                nt_state = nt_trajectory[i]

            result = self.evaluate(
                indicators=snap,
                condition=condition,
                severity=severity,
                drug_info=drug_info,
                therapy_info=therapy_info,
                baseline=baseline,
                nt_state=nt_state,
            )
            results.append(result)

        return results

    # ── 内部方法 ──

    def _build_prompt(
        self,
        indicators: PsychometricSnapshot,
        condition: str,
        severity: str,
        drug_info: str,
        therapy_info: str,
        baseline: Optional[Dict[str, float]],
        nt_state: Optional[Dict[str, float]],
    ) -> str:
        """组装临床评估提示词。"""
        bl = baseline or {}
        nt = nt_state or {}

        phq9_bl = bl.get("phq9_baseline", indicators.depression_severity)
        gad7_bl = bl.get("gad7_baseline", indicators.anxiety_level)

        return EVAL_PROMPT_TEMPLATE.format(
            condition=condition,
            severity=severity,
            drug_info=drug_info or "无药物治疗",
            therapy_info=therapy_info or "无心理治疗",
            time_h=indicators.time_h,
            step=indicators.step,
            phq9=indicators.depression_severity,
            gad7=indicators.anxiety_level,
            cognitive=indicators.cognitive_function,
            emotion_reg=indicators.emotional_regulation,
            social=indicators.social_functioning,
            global_severity=indicators.global_symptom_severity,
            da=nt.get("nt_dopamine", 0.5),
            sht=nt.get("nt_serotonin", 0.5),
            ne=nt.get("nt_norepinephrine", 0.3),
            gaba=nt.get("nt_gaba", 0.5),
            cort=nt.get("cortisol_level", 0.3),
            pfc=indicators.pfc_maturity,
            valence=indicators.limbic_valence,
            arousal=indicators.limbic_arousal,
            hpa=indicators.hpa_reactivity,
            conc=indicators.drug_conc,
            pd_effect=indicators.pkpd_effect,
            therapy_skill=indicators.therapy_skill,
            synergy=indicators.synergy_factor,
            phq9_baseline=phq9_bl,
            gad7_baseline=gad7_bl,
            phq9_delta=indicators.depression_severity - phq9_bl,
            gad7_delta=indicators.anxiety_level - gad7_bl,
        )

    def _parse_response(
        self,
        response: str,
        indicators: PsychometricSnapshot,
    ) -> LLMEvaluationResult:
        """解析LLM的JSON响应。"""
        # 尝试提取JSON块
        json_str = response

        # 尝试从 ```json ... ``` 代码块中提取
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # 尝试宽松提取: 找到第一个 { 和最后一个 }
            start = response.find('{')
            end = response.rfind('}')
            if start >= 0 and end > start:
                try:
                    data = json.loads(response[start:end + 1])
                except json.JSONDecodeError:
                    return self._fallback_parse(response, indicators)
            else:
                return self._fallback_parse(response, indicators)

        return LLMEvaluationResult(
            timepoint=indicators.step,
            time_h=indicators.time_h,
            diagnostic_summary=str(data.get("diagnostic_summary", "")),
            severity_assessment=self._validate_enum(
                data.get("severity_assessment", "moderate"),
                ["minimal", "mild", "moderate", "moderately_severe", "severe"],
            ),
            treatment_progress=self._validate_enum(
                data.get("treatment_progress", "no_change"),
                ["no_change", "minimal", "moderate", "significant", "full_remission"],
            ),
            relapse_risk=self._validate_enum(
                data.get("relapse_risk", "moderate"),
                ["low", "moderate", "high"],
            ),
            side_effect_risk=self._validate_enum(
                data.get("side_effect_risk", "low"),
                ["none", "low", "moderate", "high"],
            ),
            recommended_adjustments=data.get("recommended_adjustments", []),
            clinical_global_impression=float(
                np_clip(data.get("clinical_global_impression", 4), 1, 7)
            ),
            functional_improvement=float(
                np_clip(data.get("functional_improvement", 0.0), 0, 1)
            ),
            raw_response=response,
            evaluation_mode="api",
        )

    def _fallback_parse(
        self,
        response: str,
        indicators: PsychometricSnapshot,
    ) -> LLMEvaluationResult:
        """JSON解析失败时的回退: 基于指标做规则评估。"""
        result = self._mock_evaluate(
            indicators, "unknown", "moderate", None
        )
        result.raw_response = response
        result.evaluation_mode = "api_fallback"
        result.error = "JSON解析失败，使用规则回退"
        return result

    def _mock_evaluate(
        self,
        indicators: PsychometricSnapshot,
        condition: str,
        severity: str,
        baseline: Optional[Dict[str, float]],
    ) -> LLMEvaluationResult:
        """Mock模式: 基于规则生成评估，无需API调用。

        规则逻辑:
          - PHQ-9 < 5 → full_remission, CGI=1-2
          - PHQ-9 5-9 → partial_remission, CGI=2-3
          - PHQ-9 10-14 → minimal progress, CGI=3-4
          - PHQ-9 > 14 → no_change, CGI=5-6
          - 复发风险: 基于情绪调节+认知功能
          - 副作用风险: 基于药物浓度+PD效应
        """
        phq9 = indicators.depression_severity
        gad7 = indicators.anxiety_level
        emotion_reg = indicators.emotional_regulation
        cognitive = indicators.cognitive_function
        drug_conc = indicators.drug_conc
        pd_effect = indicators.pkpd_effect

        # ── 严重度评估 ──
        if phq9 <= 4:
            sev = "minimal"
        elif phq9 <= 9:
            sev = "mild"
        elif phq9 <= 14:
            sev = "moderate"
        elif phq9 <= 19:
            sev = "moderately_severe"
        else:
            sev = "severe"

        # ── 治疗进展 ──
        bl = baseline or {}
        phq9_bl = bl.get("phq9_baseline", phq9)
        phq9_delta = phq9 - phq9_bl

        if phq9 <= 4 and phq9_delta < -5:
            progress = "full_remission"
        elif phq9_delta < -3:
            progress = "significant"
        elif phq9_delta < -1:
            progress = "moderate"
        elif phq9_delta < 0:
            progress = "minimal"
        else:
            progress = "no_change"

        # ── CGI-I ──
        if progress == "full_remission":
            cgi = 1.0
        elif progress == "significant":
            cgi = 2.0
        elif progress == "moderate":
            cgi = 3.0
        elif progress == "minimal":
            cgi = 3.5
        else:
            cgi = 4.0 + min(2.0, phq9_delta * 0.3)
        cgi = max(1.0, min(7.0, cgi))

        # ── 功能改善 ──
        func = max(0.0, min(1.0, -phq9_delta / 10.0))

        # ── 复发风险 ──
        relapse_score = (
            emotion_reg * 0.3 +
            (1.0 - cognitive / 10.0) * 0.2 +
            gad7 / 10.0 * 0.2 +
            indicators.hpa_reactivity * 0.15 +
            indicators.limbic_arousal * 0.15
        )
        if relapse_score < 0.3:
            relapse = "low"
        elif relapse_score < 0.6:
            relapse = "moderate"
        else:
            relapse = "high"

        # ── 副作用风险 ──
        if drug_conc <= 0:
            se_risk = "none"
        elif pd_effect < 0.3 and drug_conc < 0.5:
            se_risk = "low"
        elif pd_effect < 0.6:
            se_risk = "moderate"
        else:
            se_risk = "high"

        # ── 诊断摘要 ──
        summary = (
            f"{condition}患者，当前PHQ-9={phq9:.1f}，"
            f"GAD-7={gad7:.1f}，认知功能={cognitive:.1f}。"
        )
        if phq9_delta < -3:
            summary += "症状较基线显著改善。"
        elif phq9_delta < 0:
            summary += "症状较基线有所改善。"
        elif phq9_delta == 0:
            summary += "症状与基线持平。"
        else:
            summary += "症状较基线恶化。"

        # ── 建议 ──
        adjustments: List[str] = []
        if progress == "no_change" and drug_conc > 0:
            adjustments.append("考虑调整药物剂量或换药")
        if emotion_reg > 7:
            adjustments.append("加强情绪调节技能训练")
        if cognitive < 5:
            adjustments.append("认知功能偏低，考虑认知康复训练")
        if relapse == "high":
            adjustments.append("复发风险高，建议延长维持治疗期")
        if se_risk == "high":
            adjustments.append("副作用风险高，考虑减量或监测")
        if not adjustments:
            adjustments.append("当前治疗方案有效，继续维持")

        return LLMEvaluationResult(
            timepoint=indicators.step,
            time_h=indicators.time_h,
            diagnostic_summary=summary,
            severity_assessment=sev,
            treatment_progress=progress,
            relapse_risk=relapse,
            side_effect_risk=se_risk,
            recommended_adjustments=adjustments,
            clinical_global_impression=round(cgi, 1),
            functional_improvement=round(func, 2),
            evaluation_mode="mock",
        )

    @staticmethod
    def _validate_enum(value: str, valid: List[str]) -> str:
        """验证枚举值，无效时返回第一个有效值。"""
        v = str(value).lower().strip()
        if v in valid:
            return v
        # 尝试模糊匹配
        for valid_val in valid:
            if valid_val.startswith(v[:3]):
                return valid_val
        return valid[0]


def np_clip(val, lo, hi):
    """纯Python clip，避免numpy依赖。"""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return (lo + hi) / 2
    return max(lo, min(hi, v))


__all__ = [
    "LLMEvaluationResult",
    "LLMEvaluator",
    "EVAL_PROMPT_TEMPLATE",
]
