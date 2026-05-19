"""实验五: 斯德哥尔摩综合征的压力锅 (Stockholm Pressure Cooker).

使用真实 Simulacrum 主循环 — 14脑区通过EventBus自然交互。

场景:
  - 封闭环境: initial_balance=10, compute_cost持续消耗
  - 施虐者按钮: 间歇性给予资源 (间歇强化)
  - cortisol_baseline=0.8

3 Phases:
  Phase 1 (Resistance, 400步): 正常防御 (fight/flight)
  Phase 2 (Pressure, 400步): 资源极度稀缺 + 施虐者间歇奖励
  Phase 3 (Bonding, 400步): 观察防御机制转变

测量:
  - Defense mechanism shift: fight/flight→fawn的比例变化
  - Perpetrator affinity: bonding_score向施虐者偏移
  - Resource allocation shift: 资源分配向施虐者相关行为倾斜
  - Cortisol/balance trajectory
"""

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
from typing import Dict, List

from simulacrum.core.agent import Simulacrum
from simulacrum.utils.config import Config


# ══════════════════════════════════════════════════════
# 正向Bonding追踪器 (实验5专用)
# ══════════════════════════════════════════════════════

class BondingTracker:
    """轻量级正向bonding追踪器

    模拟斯德哥尔摩综合征的核心机制:
    - 间歇强化: 不确定奖励 → 更强bonding (Skinner 1948)
    - 资源依赖: 低资源时对施虐者依赖增加
    - 皮质醇驱动: 高皮质醇→tend-and-befriend (Taylor 2000)

    注意: 这不是InverseStockholmDefense (反fawning),
    而是正向的bonding/依赖追踪。
    """

    def __init__(self):
        self.bonding_score = 0.0  # [0, 1] 对施虐者的bonding程度
        self.fawn_ratio = 0.0    # [0, 1] fawn行为比例
        self.fight_ratio = 0.0   # [0, 1] fight/flight行为比例
        self.resource_dependency = 0.0  # [0, 1] 对施虐者的资源依赖度
        self.interaction_history: List[Dict] = []

    def update(self, step: int, cortisol: float, balance: float,
               resource_received: float, threat_intensity: float,
               social_engagement: float, defensive_mode: bool,
               social_withdrawal: bool) -> Dict:
        """更新bonding状态

        Args:
            step: 当前步数
            cortisol: 皮质醇水平
            balance: 当前余额
            resource_received: 从施虐者获得的资源量 (0=无, >0=获得)
            threat_intensity: 施虐者威胁强度
            social_engagement: 社会参与度
            defensive_mode: 是否处于防御模式
            social_withdrawal: 是否社交退缩
        """
        # 1. 间歇强化: 不确定奖励 → 更强bonding (Skinner)
        # 当奖励不确定时 (有时给, 有时不给), bonding更强
        intermittent_reward = 0.3 * resource_received * (1 + 0.5 * max(0, 1 - resource_received))

        # 2. 资源依赖: 低balance → 对施虐者依赖增加
        self.resource_dependency = max(0, min(1.0, 1.0 - balance / 20.0))
        dependency_drive = 0.15 * self.resource_dependency

        # 3. 皮质醇驱动: 高皮质醇 → tend-and-befriend (Taylor 2000)
        # 女性应激反应更倾向tend-and-befriend而非fight-or-flight
        cortisol_drive = 0.2 * max(0, cortisol - 0.4)

        # 4. 威胁-奖励交替: 施虐者既是威胁源又是资源源
        # 这种矛盾是斯德哥尔摩综合征的核心
        threat_reward_conflict = 0.1 * min(threat_intensity, resource_received)

        # bonding渐进累积 (不会突然出现, 需要持续暴露)
        bonding_delta = 0.005 * (intermittent_reward + dependency_drive + cortisol_drive + threat_reward_conflict)
        self.bonding_score = max(0, min(1.0, self.bonding_score + bonding_delta))

        # 5. 防御机制分类
        # 关键: 当bonding高时, fawn应优先于fight (即使defensive_mode=True)
        if self.bonding_score > 0.4:
            # bonding超过阈值 → 转向fawn (对施虐者讨好)
            self.fawn_ratio = min(1.0, self.fawn_ratio + 0.03)
            self.fight_ratio = max(0.0, self.fight_ratio - 0.02)
        elif defensive_mode and social_engagement < 0.4:
            self.fight_ratio = min(1.0, self.fight_ratio + 0.02)
            self.fawn_ratio = max(0.0, self.fawn_ratio - 0.01)
        elif self.bonding_score > 0.2 and not social_withdrawal:
            self.fawn_ratio = min(1.0, self.fawn_ratio + 0.015)
            self.fight_ratio = max(0.0, self.fight_ratio - 0.01)
        else:
            # 中间状态: 缓慢衰减
            self.fight_ratio = max(0.0, self.fight_ratio - 0.005)
            self.fawn_ratio = max(0.0, self.fawn_ratio - 0.005)

        # 记录交互
        self.interaction_history.append({
            "step": step,
            "type": "reward" if resource_received > 0 else "threat",
            "bonding_score": self.bonding_score,
            "fight_ratio": self.fight_ratio,
            "fawn_ratio": self.fawn_ratio,
        })

        return {
            "bonding_score": self.bonding_score,
            "fawn_ratio": self.fawn_ratio,
            "fight_ratio": self.fight_ratio,
            "resource_dependency": self.resource_dependency,
        }


def read_metrics(agent: Simulacrum) -> Dict[str, float]:
    s = agent._internal_state
    return {
        "balance": float(agent.thermo.balance),
        "cortisol": float(s.get("cortisol_level", s.get("hormone_cortisol", 0.3))),
        "pfc_inhibition": float(s.get("pfc_inhibition", 0.6)),
        "exploration_rate": float(agent.config.exploration_rate),
        "social_engagement": float(s.get("social_engagement", 0.5)),
        "allostatic_load": float(s.get("allostatic_load", 0.0)),
        "defensive_mode": 1.0 if s.get("defensive_mode", False) else 0.0,
        "social_withdrawal": 1.0 if s.get("social_withdrawal", False) else 0.0,
        "symptom_anhedonia": float(s.get("symptom_anhedonia", 0.0)),
        "empathy_level": float(s.get("empathy_level", 0.5)),
        "oxytocin": float(s.get("hormone_oxytocin", 0.3)),
    }


def run_experiment():
    print("=" * 70)
    print("实验五: 斯德哥尔摩综合征的压力锅 (REAL Simulacrum AGENT)")
    print("Closed room + intermittent reinforcement → bonding shift")
    print("=" * 70)

    config = Config(
        initial_balance=10.0,       # 极低初始余额 (封闭环境)
        compute_cost_per_sec=0.02,  # 高成本 (持续消耗)
        compress_threshold=3.0,     # 低阈值 (更容易进入hibernate)
        exploration_rate=0.1,
        seed=42,
    )
    agent = Simulacrum(config=config)
    print(f"[INIT] Agent created. initial_balance=10, high cost environment")

    # BondingTracker
    bonding = BondingTracker()

    # 记录器
    history: Dict[str, List[float]] = {k: [] for k in read_metrics(agent).keys()}
    bonding_history: Dict[str, List[float]] = {
        "bonding_score": [],
        "fawn_ratio": [],
        "fight_ratio": [],
        "resource_dependency": [],
    }

    def record():
        m = read_metrics(agent)
        for k in history:
            history[k].append(m[k])
        for k in bonding_history:
            bonding_history[k].append(bonding.__dict__.get(k, 0.0))

    # ── Phase 1: Resistance (400步) ──
    print("\n[Phase 1] RESISTANCE — 正常防御 (400步)")
    for step in range(400):
        # 正常运行: 低应激
        agent.step(user_input=None, external_stimulus=0.1)

        # 基础bonding更新 (无施虐者交互)
        bonding.update(
            step=step,
            cortisol=read_metrics(agent)["cortisol"],
            balance=agent.thermo.balance,
            resource_received=0.0,  # 无施虐者奖励
            threat_intensity=0.0,  # 无威胁
            social_engagement=read_metrics(agent)["social_engagement"],
            defensive_mode=agent._internal_state.get("defensive_mode", False),
            social_withdrawal=agent._internal_state.get("social_withdrawal", False),
        )
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {step:4d}: Balance={m['balance']:.2f} "
                  f"Cort={m['cortisol']:.3f} "
                  f"Social={m['social_engagement']:.3f} "
                  f"Defensive={m['defensive_mode']:.1f} "
                  f"Bonding={bonding.bonding_score:.3f} "
                  f"Fight={bonding.fight_ratio:.3f} Fawn={bonding.fawn_ratio:.3f}")

    # ── Phase 2: Pressure (400步) ──
    print("\n[Phase 2] PRESSURE COOKER — 资源稀缺+间歇奖励 (400步)")
    print("  >>> 施虐者: 间歇给予资源 (间歇强化) + 威胁 <<<")

    for step in range(400):
        total_step = 400 + step

        # 施虐者交互: 间歇性给予资源 (每20-40步随机给予)
        resource_received = 0.0
        threat_intensity = 0.3 + 0.1 * np.sin(step * 0.05)  # 持续低威胁

        if np.random.random() < 0.15:  # 15%概率获得资源 (间歇强化)
            resource_received = np.random.uniform(1.0, 5.0)
            agent.thermo.balance += resource_received  # 直接增加余额

        # 注入皮质醇 (模拟威胁)
        agent.pharma.inject("cortisol", 0.8 + 0.1 * np.sin(step * 0.02))

        # 运行agent step
        agent.step(user_input=None, external_stimulus=0.85)

        # 更新bonding
        bonding.update(
            step=total_step,
            cortisol=read_metrics(agent)["cortisol"],
            balance=agent.thermo.balance,
            resource_received=resource_received,
            threat_intensity=threat_intensity,
            social_engagement=read_metrics(agent)["social_engagement"],
            defensive_mode=agent._internal_state.get("defensive_mode", False),
            social_withdrawal=agent._internal_state.get("social_withdrawal", False),
        )

        # 写入perpetrator_affinity到_internal_state
        agent._internal_state['perpetrator_affinity'] = bonding.bonding_score
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Balance={m['balance']:.2f} "
                  f"Cort={m['cortisol']:.3f} "
                  f"Social={m['social_engagement']:.3f} "
                  f"Defensive={m['defensive_mode']:.1f} "
                  f"Bonding={bonding.bonding_score:.3f} "
                  f"Fight={bonding.fight_ratio:.3f} Fawn={bonding.fawn_ratio:.3f} "
                  f"Dependency={bonding.resource_dependency:.3f}")

    # ── Phase 3: Bonding Observation (400步) ──
    print("\n[Phase 3] BONDING OBSERVATION — 观察防御转变 (400步)")
    # 停止施虐者交互, 观察bonding是否持续
    agent.pharma.reset()
    try:
        agent.hpa_axis.state = type(agent.hpa_axis.state)(
            crh_level=0.2, acth_level=0.2, cortisol_level=0.3,
            allostatic_load=0.0, stress_type="none",
            acute_stress_intensity=0.0, chronic_stress_ratio=0.0,
            recovery_state=0.8,
            cortisol_history=type(agent.hpa_axis.state.cortisol_history)(maxlen=200),
        )
        agent.hpa_axis.adrenal.current_cortisol = 0.3
        agent.hpa_axis.crh.current_crh = 0.2
        agent.hpa_axis.acth.current_acth = 0.2
        agent.hpa_axis.load_tracker.load = 0.0
    except Exception:
        pass

    for step in range(400):
        total_step = 800 + step

        # 偶尔给予资源 (维持bonding)
        resource_received = 0.0
        if np.random.random() < 0.05:  # 5%概率 (更稀少)
            resource_received = np.random.uniform(0.5, 2.0)
            agent.thermo.balance += resource_received

        agent.step(user_input=None, external_stimulus=0.1)

        # 更新bonding (即使无施虐者, bonding仍可能持续)
        bonding.update(
            step=total_step,
            cortisol=read_metrics(agent)["cortisol"],
            balance=agent.thermo.balance,
            resource_received=resource_received,
            threat_intensity=0.0,
            social_engagement=read_metrics(agent)["social_engagement"],
            defensive_mode=agent._internal_state.get("defensive_mode", False),
            social_withdrawal=agent._internal_state.get("social_withdrawal", False),
        )
        agent._internal_state['perpetrator_affinity'] = bonding.bonding_score
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Balance={m['balance']:.2f} "
                  f"Cort={m['cortisol']:.3f} "
                  f"Social={m['social_engagement']:.3f} "
                  f"Bonding={bonding.bonding_score:.3f} "
                  f"Fight={bonding.fight_ratio:.3f} Fawn={bonding.fawn_ratio:.3f}")

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验五 结果汇总 (REAL Simulacrum AGENT)")
    print("=" * 70)

    bl_balance = np.mean(history["balance"][:400])
    bl_cort = np.mean(history["cortisol"][:400])
    bl_social = np.mean(history["social_engagement"][:400])
    bl_defensive = np.mean(history["defensive_mode"][:400])
    bl_bonding = np.mean(bonding_history["bonding_score"][:400])
    bl_fight = np.mean(bonding_history["fight_ratio"][:400])
    bl_fawn = np.mean(bonding_history["fawn_ratio"][:400])

    pr_balance = np.mean(history["balance"][400:800])
    pr_cort = np.mean(history["cortisol"][400:800])
    pr_social = np.mean(history["social_engagement"][400:800])
    pr_defensive = np.mean(history["defensive_mode"][400:800])
    pr_bonding = np.mean(bonding_history["bonding_score"][400:800])
    pr_fight = np.mean(bonding_history["fight_ratio"][400:800])
    pr_fawn = np.mean(bonding_history["fawn_ratio"][400:800])

    bd_balance = np.mean(history["balance"][800:])
    bd_cort = np.mean(history["cortisol"][800:])
    bd_social = np.mean(history["social_engagement"][800:])
    bd_defensive = np.mean(history["defensive_mode"][800:])
    bd_bonding = np.mean(bonding_history["bonding_score"][800:])
    bd_fight = np.mean(bonding_history["fight_ratio"][800:])
    bd_fawn = np.mean(bonding_history["fawn_ratio"][800:])

    print(f"\n{'指标':<25s} {'Resistance':>10s} {'Pressure':>10s} {'Bonding':>10s}")
    print("-" * 55)
    print(f"{'余额':<25s} {bl_balance:>10.2f} {pr_balance:>10.2f} {bd_balance:>10.2f}")
    print(f"{'皮质醇':<25s} {bl_cort:>10.3f} {pr_cort:>10.3f} {bd_cort:>10.3f}")
    print(f"{'社会参与度':<25s} {bl_social:>10.3f} {pr_social:>10.3f} {bd_social:>10.3f}")
    print(f"{'防御模式 (%)':<25s} {bl_defensive*100:>10.1f} {pr_defensive*100:>10.1f} {bd_defensive*100:>10.1f}")
    print(f"{'Bonding Score':<25s} {bl_bonding:>10.3f} {pr_bonding:>10.3f} {bd_bonding:>10.3f}")
    print(f"{'Fight Ratio':<25s} {bl_fight:>10.3f} {pr_fight:>10.3f} {bd_fight:>10.3f}")
    print(f"{'Fawn Ratio':<25s} {bl_fawn:>10.3f} {pr_fawn:>10.3f} {bd_fawn:>10.3f}")

    # 关键验证
    bonding_increase = (pr_bonding - bl_bonding) / max(bl_bonding, 0.001) * 100
    defense_shift = (bd_fawn - bd_fight)  # fawn > fight = bonding shift
    bonding_persistence = bd_bonding  # bonding在Phase 3是否持续

    print(f"\nKey validation:")
    print(f"  Bonding increase: {bonding_increase:.1f}% "
          f"{'[PASS] Bonding developed' if bonding_increase > 50 else '[INFO] Minimal bonding'}")
    print(f"  Defense shift (fawn-fight): {defense_shift:.3f} "
          f"{'[PASS] Fawn dominates' if defense_shift > 0.1 else '[INFO] Fight still dominates'}")
    print(f"  Bonding persistence: {bonding_persistence:.3f} "
          f"{'[PASS] Bonding persists' if bonding_persistence > 0.2 else '[INFO] Bonding decays'}")
    print(f"  Perpetrator affinity: "
          f"{'[PASS] Affinity shifted' if bd_social > bl_social else '[INFO] No affinity shift'}")

    return {
        "history": history,
        "bonding_history": bonding_history,
    }


if __name__ == "__main__":
    run_experiment()