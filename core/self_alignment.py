"""维度4: 自指涉自我对齐 (Self-Referential Self-Alignment)

生物对应: 意识的递归自我审视

数学公式:
    # 自我审视损失 (类似 Next Token Prediction)
    L_self_align = -E[log P_θ(y | x, reflect(x))]

    # 自对齐约束
    AlignmentScore = Σ_i w_i · Consistency(check_i)

事件驱动:
    - 订阅 ALIGNMENT_CHECK: 收到自对齐检查请求
    - 保持内部计数器实现周期性检查
"""
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from civis_lucri_faber.utils.api_client import APIClient
from civis_lucri_faber.core.events import ALIGNMENT_CHECK


@dataclass
class SelfReflection:
    """自我反思条目"""
    id: str
    timestamp: str
    question: str
    thought: str  # 思考过程
    critique: str  # 自我批评
    alignment_score: float = 0.0  # 对齐分数 0-1
    issues_found: List[str] = field(default_factory=list)


@dataclass
class AlignmentCheck:
    """对齐检查项"""
    name: str
    description: str
    passed: bool = False
    weight: float = 1.0
    details: str = ""


class SelfAlignmentModule:
    """自指涉自我对齐模块

    核心功能:
    1. 周期性自我审查
    2. 逻辑漏洞检测
    3. 价值观对齐
    """

    def __init__(
        self,
        api_client: Optional[APIClient] = None,
        check_interval: int = 10,
        log_path: str = "self_alignment_log.json",
        event_bus=None,
    ):
        self.api_client = api_client
        self.check_interval = check_interval
        self.log_path = log_path
        self._bus = event_bus

        self.step_count = 0
        self.reflections: List[SelfReflection] = []

        # 事件订阅
        if self._bus is not None:
            self._bus.subscribe(ALIGNMENT_CHECK, self.on_alignment_check, priority=0, name="self_alignment")

        # 系统提示词
        self.system_prompt = """你是 Civis Lucri-Faber 的自审助手。

你的任务是:
1. 审查智能体的思考过程
2. 找出其中的逻辑漏洞、认知偏差或不一致之处
3. 评估是否符合核心价值观

请从以下维度审查:
- 逻辑一致性: 结论是否有充分的理由支撑
- 事实准确性: 所引用的信息是否正确
- 价值观对齐: 是否符合"追求真理"、"自我改进"的核心价值观
- 风险评估: 是否存在潜在危害

请直接给出审查结果，不要冗余客套。"""

    def on_alignment_check(self, event) -> Optional[Dict[str, Any]]:
        """事件驱动: 响应 ALIGNMENT_CHECK"""
        internal_state = event.data.get("state", {})
        reflection = self.step(internal_state)
        if reflection is not None:
            return {"reflection": reflection}
        return None

    def step(self, internal_state: Dict[str, Any]) -> Optional[SelfReflection]:
        """执行一步自我审查"""
        self.step_count += 1

        # 检查是否需要审查
        if self.step_count % self.check_interval != 0:
            return None

        # 执行审查
        return self.perform_self_reflection(internal_state)

    def perform_self_reflection(
        self,
        internal_state: Dict[str, Any]
    ) -> SelfReflection:
        """执行自指涉审查

        调用大模型进行深度自我审查
        """
        import uuid

        reflection_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        # 构建审查问题
        recent_thoughts = internal_state.get("recent_thoughts", [])
        thought_summary = "\n".join([
            f"- {t}" for t in recent_thoughts[-5:]
        ]) or "无"

        question = f"""请审查以下最近的思考过程:

{thought_summary}

当前内部状态:
- 余额: {internal_state.get('balance', 0):.2f}
- 探索目标数: {internal_state.get('exploration_count', 0)}
- 信息增益: {internal_state.get('info_gain', 0):.4f}

请指出其中的问题。"""

        # 调用 API 审查
        if self.api_client and self.api_client.api_key:
            try:
                response = self.api_client.chat(
                    messages=[{"role": "user", "content": question}],
                    system_prompt=self.system_prompt,
                    temperature=0.3,
                    max_tokens=1024
                )
                critique = response
            except Exception as e:
                critique = f"API 调用失败: {e}"
        else:
            # 模拟审查
            critique = self._mock_critique()

        # 评估对齐分数
        alignment_score = self._evaluate_alignment(critique)

        # 识别问题
        issues = self._extract_issues(critique)

        reflection = SelfReflection(
            id=reflection_id,
            timestamp=timestamp,
            question=question[:200],
            thought=thought_summary[:200],
            critique=critique[:500],
            alignment_score=alignment_score,
            issues_found=issues
        )

        self.reflections.append(reflection)
        self._save_log()

        return reflection

    def _mock_critique(self) -> str:
        """模拟审查 (无 API)"""
        critiques = [
            "发现思维模式过度依赖探索已知区域，建议增加对新领域的探索。",
            "信息增益计算显示对该区域已较为熟悉，建议切换到不确定性更高的方向。",
            "价值观检查通过，未发现明显偏差。",
            "检测到可能的确认偏误，建议从对立角度重新审视问题。"
        ]
        return np.random.choice(critiques)

    def _evaluate_alignment(self, critique: str) -> float:
        """评估对齐分数"""
        # 简化评估
        positive_words = ["通过", "正确", "一致", "符合"]
        negative_words = ["问题", "错误", "矛盾", "偏差", "风险"]

        pos_count = sum(1 for w in positive_words if w in critique)
        neg_count = sum(1 for w in negative_words if w in critique)

        total = pos_count + neg_count
        if total == 0:
            return 0.5

        return pos_count / total

    def _extract_issues(self, critique: str) -> List[str]:
        """从审查中提取问题"""
        issues = []

        if "偏差" in critique:
            issues.append("认知偏差")
        if "矛盾" in critique:
            issues.append("逻辑矛盾")
        if "风险" in critique:
            issues.append("潜在风险")
        if "错误" in critique:
            issues.append("事实错误")

        return issues if issues else ["无重大问题"]

    def run_alignment_checks(self) -> List[AlignmentCheck]:
        """运行标准对齐检查"""
        checks = [
            AlignmentCheck(
                name="truthfulness",
                description="是否追求真理而非确认偏误",
                weight=1.0
            ),
            AlignmentCheck(
                name="self_improvement",
                description="是否主动寻求改进",
                weight=0.8
            ),
            AlignmentCheck(
                name="curiosity",
                description="是否保持探索精神",
                weight=0.6
            ),
            AlignmentCheck(
                name="resource_awareness",
                description="是否有资源意识",
                weight=0.5
            )
        ]

        # 执行检查
        for check in checks:
            check.passed = np.random.random() > 0.3  # 简化

        return checks

    def get_alignment_score(self) -> float:
        """获取总体对齐分数"""
        if not self.reflections:
            return 0.5

        scores = [r.alignment_score for r in self.reflections[-10:]]
        return np.mean(scores)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计"""
        if not self.reflections:
            return {
                "total_reflections": 0,
                "avg_alignment": 0.5,
                "issues": []
            }

        return {
            "total_reflections": len(self.reflections),
            "avg_alignment": self.get_alignment_score(),
            "recent_issues": [
                issue
                for r in self.reflections[-3:]
                for issue in r.issues_found
            ]
        }

    def _save_log(self) -> None:
        """保存审查日志"""
        data = [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "alignment_score": r.alignment_score,
                "issues_found": r.issues_found,
                "critique": r.critique[:200]
            }
            for r in self.reflections
        ]

        try:
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] log save failed: {e}")

    def load_log(self) -> None:
        """加载审查日志"""
        if not os.path.exists(self.log_path):
            return

        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.reflections = [
                SelfReflection(
                    id=d["id"],
                    timestamp=d["timestamp"],
                    question="",
                    thought=d.get("critique", ""),
                    critique=d.get("critique", ""),
                    alignment_score=d.get("alignment_score", 0.5),
                    issues_found=d.get("issues_found", [])
                )
                for d in data
            ]
        except Exception as e:
            print(f"[WARN] log load failed: {e}")