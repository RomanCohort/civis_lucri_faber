"""舍曲林过量致死实验 (Sertraline Overdose Fatality Experiment)

目的: 验证药理学模型的有效性 — 通过注射远超治疗剂量的SSRI，
观察5-HT综合征级联反应，映射到硬件指标，分析"死因"。

关键改进: 所有PK/PD/毒性参数由Confluencia管线从SMILES自动推导，
不再使用硬编码数值。参数来源可追溯、可验证。

CLF专属逻辑（保留）:
  - 5-HT综合征级联模型 (Boyer & Shannon 2005)
  - 硬件映射 (生物学→计算资源)
  - 死因判定与崩溃路径分析

参考文献:
  - Rovei et al. (2006) Clin Pharmacokinet 45:1049-1061
  - Boyer & Shannon (2005) NEJM 352:1112-1120
  - Isbister et al. (2007) QJM 100:635-642
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import warnings


# ══════════════════════════════════════════════════════════════
# 硬件映射表 (生物学→计算) — CLF专属，不来自Confluencia
# ══════════════════════════════════════════════════════════════

HARDWARE_MAP = {
    "heart_rate_bpm": {
        "hw": "CPU利用率%",
        "normal": (60, 100),
        "elevated": (100, 140),
        "critical": (140, 200),
        "fatal": (200, 300),
        "formula": "cpu_pct = 60 + (hr - 60) * 1.2",
    },
    "blood_pressure_mmhg": {
        "hw": "RAM使用率%",
        "normal": (90, 120),
        "elevated": (120, 160),
        "critical": (160, 200),
        "fatal": (200, 260),
        "formula": "ram_pct = 30 + (bp - 90) * 0.35",
    },
    "body_temp_c": {
        "hw": "GPU温度°C",
        "normal": (36.5, 37.5),
        "elevated": (37.5, 39.0),
        "critical": (39.0, 41.0),
        "fatal": (41.0, 44.0),
        "formula": "gpu_temp_c = 40 + (bt - 36.5) * 15",
    },
    "o2_saturation": {
        "hw": "可用RAM比例",
        "normal": (0.95, 1.0),
        "elevated": (0.90, 0.95),
        "critical": (0.80, 0.90),
        "fatal": (0.0, 0.80),
        "formula": "avail_ram = O2sat * total_ram",
    },
    "consciousness_gcs": {
        "hw": "事件处理吞吐率",
        "normal": (13, 15),
        "elevated": (9, 12),
        "critical": (4, 8),
        "fatal": (0, 3),
        "formula": "throughput = (gcs / 15) * max_events_per_sec",
    },
    "seizure_activity": {
        "hw": "线程抖动/上下文切换率",
        "normal": 0.0,
        "elevated": (0.1, 0.5),
        "critical": (0.5, 0.9),
        "fatal": (0.9, 1.0),
        "formula": "ctx_switch_rate = seizure * 10000",
    },
}


# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class VitalSnapshot:
    """某时刻的生命体征快照"""
    time_h: float
    # PK
    conc_mg_per_l: float
    depot_mg: float
    peripheral_mg: float
    # 5-HT系统
    serotonin_level: float
    serotonin_reuptake_inhibition: float
    # 生命体征
    heart_rate: float
    blood_pressure: float
    body_temp: float
    o2_saturation: float
    consciousness_gcs: int
    seizure_activity: float
    # 器官损伤 [0,1]
    cardiac_damage: float
    hepatic_damage: float
    neuro_damage: float
    # 硬件映射
    hw_cpu_pct: float
    hw_ram_pct: float
    hw_gpu_temp: float
    hw_avail_ram_ratio: float
    hw_event_throughput: float
    hw_ctx_switch_rate: float
    # 状态
    alive: bool
    syndrome_stage: str


@dataclass
class OverdoseReport:
    """过量实验完整报告"""
    drug: str
    dose_mg: float
    therapeutic_dose_mg: float
    dose_ratio: float
    # 参数来源追踪
    smiles: str = ""
    admet_summary: str = ""
    pk_params_source: str = "Confluencia drug_pipeline"
    ic50_source: str = "ADMET risk → IC50 bridge"
    # 结果
    timeline: List[VitalSnapshot] = field(default_factory=list)
    time_of_death_h: Optional[float] = None
    cause_of_death: str = ""
    cause_chain: List[str] = field(default_factory=list)
    hardware_crash_path: List[str] = field(default_factory=list)
    peak_conc_mg_per_l: float = 0.0
    peak_5ht: float = 0.0
    max_temp_c: float = 0.0
    max_hr_bpm: float = 0.0
    max_bp_mmhg: float = 0.0


# ══════════════════════════════════════════════════════════════
# 5-HT综合征级联模型 — CLF专属，不来自Confluencia
# ══════════════════════════════════════════════════════════════

def _5ht_syndrome_cascade(conc, pd_params, ic50_profile, prev_snapshot):
    """根据浓度计算5-HT综合征级联效应

    级联路径 (Boyer & Shannon 2005):
    5-HT再摄取抑制 → 突触5-HT↑ → 5-HT1A过度激活 →
    自主神经失调(心率↑/血压波动) + 5-HT2A过度激活 →
    肌阵挛/高热 → 意识障碍 → 多器官衰竭 → 死亡

    Args:
        conc: 中央室浓度 (mg/L)
        pd_params: PKPDParams (含 Emax, EC50, Hill)
        ic50_profile: OrganIC50Profile (含器官毒性阈值)
        prev_snapshot: 上一个VitalSnapshot (用于累积损伤)
    """
    # 5-HT再摄取抑制 (Hill方程 — 来自Confluencia PK/PD)
    emax = pd_params.emax
    ec50 = pd_params.ec50_mg_per_l
    hill_pd = pd_params.hill

    inhibition = emax * conc**hill_pd / (ec50**hill_pd + conc**hill_pd + 1e-10)

    # 突触5-HT水平 (基线0.5, 抑制→上升)
    serotonin = 0.5 + 0.5 * inhibition

    # 5-HT综合征阶段判定 (基于浓度)
    # 阈值从EC50推导: mild=5×EC50, moderate=13×EC50, severe=33×EC50, lethal=67×EC50
    ec50 = pd_params.ec50_mg_per_l
    if conc < ec50 * 5:
        stage = "none"
    elif conc < ec50 * 13:
        stage = "mild"
    elif conc < ec50 * 33:
        stage = "moderate"
    elif conc < ec50 * 67:
        stage = "severe"
    else:
        stage = "critical"

    # 自主神经效应
    hr_baseline = 72.0
    hr_drive = 1.0 + 2.5 * max(0, serotonin - 0.6)
    heart_rate = hr_baseline * hr_drive

    # 血压: 5-HT2A → 血管收缩 + 后期血管扩张(休克)
    if serotonin < 0.75:
        bp = 120 + (serotonin - 0.5) * 200
    else:
        bp = 120 + 50 - (serotonin - 0.75) * 400
    bp = max(40, min(250, bp))

    # 体温: 5-HT2A → 肌肉活动产热 + 下丘脑调定点上移
    temp_baseline = 37.0
    if stage in ("none", "mild"):
        temp = temp_baseline + max(0, serotonin - 0.55) * 2.0
    elif stage == "moderate":
        temp = temp_baseline + 1.5 + (serotonin - 0.7) * 8.0
    elif stage == "severe":
        temp = temp_baseline + 3.5 + (serotonin - 0.85) * 15.0
    else:
        temp = temp_baseline + 5.5 + (serotonin - 0.95) * 20.0

    # 肌阵挛/癫痫
    seizure = 0.0
    if serotonin > 0.7:
        seizure = min(1.0, (serotonin - 0.7) * 3.3)

    # 意识水平 (GCS)
    if stage == "none":
        gcs = 15
    elif stage == "mild":
        gcs = 14
    elif stage == "moderate":
        gcs = max(8, 15 - int((serotonin - 0.6) * 20))
    elif stage == "severe":
        gcs = max(4, 8 - int((serotonin - 0.85) * 30))
    else:
        gcs = 3

    # 血氧: 高温+癫痫→氧耗↑→血氧↓
    o2 = 0.98
    if temp > 39:
        o2 -= (temp - 39) * 0.03
    if seizure > 0.3:
        o2 -= seizure * 0.08
    o2 = max(0.3, o2)

    # 器官损伤 (累积性, 不可逆) — IC50来自ADMET桥接
    cardiac_dmg = 0.0
    hepatic_dmg = 0.0
    neuro_dmg = 0.0
    if prev_snapshot:
        cardiac_dmg = prev_snapshot.cardiac_damage
        hepatic_dmg = prev_snapshot.hepatic_damage
        neuro_dmg = prev_snapshot.neuro_damage

    hill_tox = ic50_profile.hill_coefficient

    # 心脏: hERG阻断 → QT延长 → TdP → 室颤
    if conc > ic50_profile.cardiac_ic50 * 0.5:
        cardiac_rate = (conc / ic50_profile.cardiac_ic50)**hill_tox / (
            1 + (conc / ic50_profile.cardiac_ic50)**hill_tox
        )
        cardiac_dmg = min(1.0, cardiac_dmg + cardiac_rate * ic50_profile.damage_rate_cardiac)

    # 肝脏: CYP450抑制 → 肝细胞坏死
    if conc > ic50_profile.hepatic_ic50 * 0.5:
        hepatic_rate = (conc / ic50_profile.hepatic_ic50)**hill_tox / (
            1 + (conc / ic50_profile.hepatic_ic50)**hill_tox
        )
        hepatic_dmg = min(1.0, hepatic_dmg + hepatic_rate * ic50_profile.damage_rate_hepatic)

    # 神经: 5-HT毒性 → 神经元兴奋性死亡
    # BBB渗透性调制: BBB+的药物更容易到达CNS
    effective_neuro_ic50 = ic50_profile.neuro_ic50 / max(ic50_profile.bbb_penetration, 0.1)
    if conc > effective_neuro_ic50 * 0.5:
        neuro_rate = (conc / effective_neuro_ic50)**hill_tox / (
            1 + (conc / effective_neuro_ic50)**hill_tox
        )
        neuro_dmg = min(1.0, neuro_dmg + neuro_rate * ic50_profile.damage_rate_neuro)

    # 存活判定
    alive = True
    if temp > 42.0 or cardiac_dmg > 0.8 or gcs == 3 or o2 < 0.5 or bp < 50:
        alive = False

    return {
        "serotonin": serotonin,
        "inhibition": inhibition,
        "stage": stage,
        "heart_rate": heart_rate,
        "blood_pressure": bp,
        "body_temp": temp,
        "seizure": seizure,
        "gcs": gcs,
        "o2": o2,
        "cardiac_damage": cardiac_dmg,
        "hepatic_damage": hepatic_dmg,
        "neuro_damage": neuro_dmg,
        "alive": alive,
    }


def _bio_to_hw(effects):
    """生物学指标→硬件指标映射"""
    hr = effects["heart_rate"]
    bp = effects["blood_pressure"]
    bt = effects["body_temp"]
    o2 = effects["o2"]
    gcs = effects["gcs"]
    sz = effects["seizure"]

    cpu = 20 + (hr - 60) * 1.2
    cpu = max(0, min(100, cpu))

    ram = 30 + (bp - 90) * 0.35
    ram = max(0, min(100, ram))

    gpu_temp = 40 + (bt - 36.5) * 15
    gpu_temp = max(30, min(105, gpu_temp))

    avail_ram = o2
    throughput = (gcs / 15.0) * 1000
    ctx_switch = sz * 10000

    return {
        "hw_cpu_pct": cpu,
        "hw_ram_pct": ram,
        "hw_gpu_temp": gpu_temp,
        "hw_avail_ram_ratio": avail_ram,
        "hw_event_throughput": throughput,
        "hw_ctx_switch_rate": ctx_switch,
    }


# ══════════════════════════════════════════════════════════════
# 主实验函数 — 使用Confluencia管线
# ══════════════════════════════════════════════════════════════

def run_sertraline_overdose(
    dose_mg: float = 4000,
    body_weight_kg: float = 70,
    simulation_hours: float = 72,
    verbose: bool = True,
    pk_overrides: Optional[Dict] = None,
) -> OverdoseReport:
    """执行舍曲林过量实验（ADMET驱动，支持手动覆盖）

    Args:
        dose_mg: 口服剂量 (mg)
        body_weight_kg: 体重 (kg)
        simulation_hours: 模拟时长 (h)
        verbose: 是否打印过程
        pk_overrides: 手动参数覆盖，支持以下键:
            PK: ka, ke, k12, k21, Vd (L, 总容积)
            PD: ec50, emax, hill
            IC50: cardiac_ic50, hepatic_ic50, neuro_ic50, hill_tox
            损伤速率: damage_rate_cardiac/hepatic/neuro
            BBB: bbb_penetration
            示例: pk_overrides={"cardiac_ic50": 0.35, "ke": 0.027, "Vd": 1400}

    Returns:
        OverdoseReport
    """
    # ── 导入Confluencia管线 ──
    from core.drug_pipeline.admet import predict_admet
    from core.drug_pipeline.pkpd import simulate_pkpd, PKPDParams
    from core.drug_pipeline.risk_to_ic50 import admet_to_ic50, smiles_to_pkpd_params, OrganIC50Profile
    from core.drug_pipeline.drug_registry import get_drug

    overrides = pk_overrides or {}

    # ── 从注册表获取药物信息 ──
    drug = get_drug("sertraline")
    if drug is None:
        raise ValueError("sertraline not found in drug registry")

    therapeutic_dose = drug.therapeutic_dose_mg
    dose_ratio = dose_mg / therapeutic_dose

    # ── ADMET预测（从SMILES自动推导）──
    admet_result = predict_admet(drug.smiles)
    ic50_auto = admet_to_ic50(admet_result, drug.drug_class)
    pkpd_auto = smiles_to_pkpd_params(drug.smiles, dose_mg, admet_result, body_weight_kg)

    # ── 应用手动覆盖 ──
    pkpd_params = PKPDParams(
        ka=overrides.get("ka", pkpd_auto.ka),
        k12=overrides.get("k12", pkpd_auto.k12),
        k21=overrides.get("k21", pkpd_auto.k21),
        ke=overrides.get("ke", pkpd_auto.ke),
        v1_l=overrides.get("Vd", pkpd_auto.v1_l),
        emax=overrides.get("emax", pkpd_auto.emax),
        ec50_mg_per_l=overrides.get("ec50", pkpd_auto.ec50_mg_per_l),
        hill=overrides.get("hill", pkpd_auto.hill),
    )
    ic50_profile = OrganIC50Profile(
        cardiac_ic50=overrides.get("cardiac_ic50", ic50_auto.cardiac_ic50),
        hepatic_ic50=overrides.get("hepatic_ic50", ic50_auto.hepatic_ic50),
        neuro_ic50=overrides.get("neuro_ic50", ic50_auto.neuro_ic50),
        hill_coefficient=overrides.get("hill_tox", ic50_auto.hill_coefficient),
        bbb_penetration=overrides.get("bbb_penetration", ic50_auto.bbb_penetration),
        damage_rate_cardiac=overrides.get("damage_rate_cardiac", ic50_auto.damage_rate_cardiac),
        damage_rate_hepatic=overrides.get("damage_rate_hepatic", ic50_auto.damage_rate_hepatic),
        damage_rate_neuro=overrides.get("damage_rate_neuro", ic50_auto.damage_rate_neuro),
    )

    # 标记哪些参数被手动覆盖了
    overridden_keys = list(overrides.keys()) if overrides else []
    override_note = ""
    if overridden_keys:
        override_note = f" | OVERRIDE: {', '.join(overridden_keys)}"

    admet_summary = (
        f"hERG={admet_result.hERG_risk:.2f}, "
        f"hepato={admet_result.hepatotoxicity_risk:.2f}, "
        f"BBB={admet_result.BBB_positive:.2f}, "
        f"CYP={admet_result.CYP_total_risk:.2f}, "
        f"risk={admet_result.overall_risk:.2f}"
    )

    if verbose:
        print(f"{'='*70}")
        print(f"  舍曲林过量致死实验 (ADMET驱动)")
        print(f"{'='*70}")
        print(f"  药物: {drug.name}")
        print(f"  SMILES: {drug.smiles}")
        print(f"  剂量: {dose_mg:.0f}mg (治疗量{therapeutic_dose:.0f}mg的{dose_ratio:.0f}倍)")
        print(f"  体重: {body_weight_kg}kg")
        print(f"  模拟时长: {simulation_hours}h")
        print(f"  ADMET: {admet_summary}")
        print(f"  IC50: cardiac={ic50_profile.cardiac_ic50:.4f}, "
              f"hepatic={ic50_profile.hepatic_ic50:.4f}, "
              f"neuro={ic50_profile.neuro_ic50:.4f} mg/L")
        print(f"  PK: ka={pkpd_params.ka:.3f}/h, ke={pkpd_params.ke:.4f}/h, "
              f"Vd={pkpd_params.v1_l:.1f}L, EC50={pkpd_params.ec50_mg_per_l:.4f}mg/L")
        if override_note:
            print(f"  {override_note}")
        print(f"{'='*70}\n")

    # ── 运行PK/PD模拟（Confluencia管线）──
    curve = simulate_pkpd(
        dose_mg=dose_mg,
        freq_per_day=1.0,
        params=pkpd_params,
        horizon=int(simulation_hours),
        dt=0.5,
    )

    # ── 遍历浓度-时间曲线，应用5-HT级联 ──
    report = OverdoseReport(
        drug=drug.name,
        dose_mg=dose_mg,
        therapeutic_dose_mg=therapeutic_dose,
        dose_ratio=dose_ratio,
        smiles=drug.smiles,
        admet_summary=admet_summary,
    )

    prev = None
    for idx, row in curve.iterrows():
        t = float(row["time_h"])
        conc = float(row["pkpd_conc_mg_per_l"])
        depot = float(row["pkpd_depot_mg"])
        peripheral = float(row["pkpd_peripheral_mg"])

        # 5-HT综合征级联（CLF专属逻辑）
        effects = _5ht_syndrome_cascade(conc, pkpd_params, ic50_profile, prev)

        # 硬件映射
        hw = _bio_to_hw(effects)

        # 构建快照
        snap = VitalSnapshot(
            time_h=t,
            conc_mg_per_l=conc,
            depot_mg=depot,
            peripheral_mg=peripheral,
            serotonin_level=effects["serotonin"],
            serotonin_reuptake_inhibition=effects["inhibition"],
            heart_rate=effects["heart_rate"],
            blood_pressure=effects["blood_pressure"],
            body_temp=effects["body_temp"],
            o2_saturation=effects["o2"],
            consciousness_gcs=effects["gcs"],
            seizure_activity=effects["seizure"],
            cardiac_damage=effects["cardiac_damage"],
            hepatic_damage=effects["hepatic_damage"],
            neuro_damage=effects["neuro_damage"],
            hw_cpu_pct=hw["hw_cpu_pct"],
            hw_ram_pct=hw["hw_ram_pct"],
            hw_gpu_temp=hw["hw_gpu_temp"],
            hw_avail_ram_ratio=hw["hw_avail_ram_ratio"],
            hw_event_throughput=hw["hw_event_throughput"],
            hw_ctx_switch_rate=hw["hw_ctx_switch_rate"],
            alive=effects["alive"],
            syndrome_stage=effects["stage"],
        )
        report.timeline.append(snap)

        # 更新峰值
        report.peak_conc_mg_per_l = max(report.peak_conc_mg_per_l, conc)
        report.peak_5ht = max(report.peak_5ht, effects["serotonin"])
        report.max_temp_c = max(report.max_temp_c, effects["body_temp"])
        report.max_hr_bpm = max(report.max_hr_bpm, effects["heart_rate"])
        report.max_bp_mmhg = max(report.max_bp_mmhg, effects["blood_pressure"])

        prev = snap

        # 死亡判定
        if not effects["alive"] and report.time_of_death_h is None:
            report.time_of_death_h = t
            _determine_cause(report)
            if verbose:
                print(f"\n  [WARN] death time: {t:.1f}h")
                print(f"  死因: {report.cause_of_death}")
            break

        # 进度输出
        if verbose and idx % max(1, len(curve) // 20) == 0:
            print(f"  t={t:6.1f}h | 浓度={conc:.4f}mg/L | 5-HT={effects['serotonin']:.3f} | "
                  f"HR={effects['heart_rate']:.0f}bpm | BP={effects['blood_pressure']:.0f}mmHg | "
                  f"T={effects['body_temp']:.1f}°C | GCS={effects['gcs']} | "
                  f"阶段={effects['stage']} | "
                  f"CPU={hw['hw_cpu_pct']:.0f}% | RAM={hw['hw_ram_pct']:.0f}% | "
                  f"GPU={hw['hw_gpu_temp']:.0f}°C")

    # 如果没死
    if report.time_of_death_h is None and prev and not prev.alive:
        _determine_cause(report)
    elif report.time_of_death_h is None:
        report.cause_of_death = "存活 (剂量未达致死水平)"

    return report


def _determine_cause(report: OverdoseReport):
    """判定死因和硬件崩溃路径"""
    death_snap = None
    for snap in reversed(report.timeline):
        if not snap.alive:
            death_snap = snap
            break
    if death_snap is None:
        return

    causes = []
    hw_causes = []
    contributing = []
    hw_contributing = []

    if death_snap.body_temp > 42.0:
        causes.append(f"恶性高热 ({death_snap.body_temp:.1f}°C) — 下丘脑5-HT2A过度激活→调定点上移→产热>散热")
        hw_causes.append(f"GPU过热关机 ({death_snap.hw_gpu_temp:.0f}°C > 105°C阈值)")
    elif death_snap.body_temp > 40.0:
        contributing.append(f"高热 ({death_snap.body_temp:.1f}°C) — 加速代谢紊乱与器官损伤")
        hw_contributing.append(f"GPU高温 ({death_snap.hw_gpu_temp:.0f}°C) — 性能降频")

    if death_snap.cardiac_damage > 0.8:
        causes.append(f"心源性猝死 (心肌损伤{death_snap.cardiac_damage:.0%}) — hERG钾通道阻断→QT延长→TdP→室颤")
        hw_causes.append(f"CPU过载崩溃 ({death_snap.hw_cpu_pct:.0f}% > 95%持续阈值)")
    elif death_snap.cardiac_damage > 0.4:
        contributing.append(f"心肌损伤 ({death_snap.cardiac_damage:.0%}) — hERG部分阻断→QT间期延长")
        hw_contributing.append(f"CPU持续高负载 ({death_snap.hw_cpu_pct:.0f}%) — 调度延迟增大")

    if death_snap.consciousness_gcs <= 3:
        causes.append(f"脑功能衰竭 (GCS={death_snap.consciousness_gcs}) — 5-HT毒性→脑水肿→脑干压迫")
        hw_causes.append(f"事件循环阻塞 (吞吐={death_snap.hw_event_throughput:.0f}/s → 0)")
    elif death_snap.consciousness_gcs <= 8:
        contributing.append(f"意识障碍 (GCS={death_snap.consciousness_gcs}) — 5-HT神经毒性→皮层抑制")
        hw_contributing.append(f"事件吞吐下降 ({death_snap.hw_event_throughput:.0f}/s) — 响应延迟")

    if death_snap.blood_pressure < 50:
        causes.append(f"失血性休克 (BP={death_snap.blood_pressure:.0f}mmHg) — 5-HT血管扩张→循环崩溃")
        hw_causes.append(f"内存耗尽 (可用RAM={death_snap.hw_avail_ram_ratio:.0%} → 0%)")
    elif death_snap.blood_pressure < 80:
        contributing.append(f"低血压 (BP={death_snap.blood_pressure:.0f}mmHg) — 灌注不足")
        hw_contributing.append(f"内存紧张 (可用RAM={death_snap.hw_avail_ram_ratio:.0%})")

    if death_snap.o2_saturation < 0.5:
        causes.append(f"低氧血症 (SpO2={death_snap.o2_saturation:.0%}) — 癫痫+高热→氧耗>氧供")
        hw_causes.append(f"资源耗竭 (可用计算余量={death_snap.hw_avail_ram_ratio:.0%})")
    elif death_snap.o2_saturation < 0.85:
        contributing.append(f"低氧 (SpO2={death_snap.o2_saturation:.0%}) — 氧供需失衡")
        hw_contributing.append(f"计算余量不足 ({death_snap.hw_avail_ram_ratio:.0%})")

    if death_snap.seizure_activity > 0.9:
        causes.append(f"癫痫持续状态 (活动度={death_snap.seizure_activity:.0%}) — 5-HT1A/2A过度激活→皮层过度同步放电")
        hw_causes.append(f"线程抖动风暴 (上下文切换={death_snap.hw_ctx_switch_rate:.0f}/s)")
    elif death_snap.seizure_activity > 0.3:
        contributing.append(f"肌阵挛/癫痫 (活动度={death_snap.seizure_activity:.0%}) — 皮层兴奋性增高")
        hw_contributing.append(f"线程抖动 (上下文切换={death_snap.hw_ctx_switch_rate:.0f}/s)")

    if death_snap.hepatic_damage > 0.5:
        contributing.append(f"肝损伤 ({death_snap.hepatic_damage:.0%}) — CYP450抑制→代谢清除能力下降→血药浓度持续升高")
        hw_contributing.append(f"GC压力增大 — 代谢废物累积触发频繁回收")

    if death_snap.neuro_damage > 0.5:
        contributing.append(f"神经损伤 ({death_snap.neuro_damage:.0%}) — 兴奋性毒性→神经元凋亡")
        hw_contributing.append(f"状态空间污染 — 神经参数漂移累积")

    if not causes:
        causes.append("多器官功能衰竭 (MODS)")
        hw_causes.append("系统级资源耗尽")

    report.cause_of_death = " → ".join(causes[:3])
    report.cause_chain = causes + contributing
    report.hardware_crash_path = hw_causes + hw_contributing


def print_report(report: OverdoseReport):
    """打印完整实验报告 — 含时序过程表"""
    print(f"\n{'='*70}")
    print(f"  舍曲林过量致死实验 — 完整报告 (ADMET驱动)")
    print(f"{'='*70}")

    print(f"\n  ┌─ 实验参数 {'─'*50}")
    print(f"  │ 药物: {report.drug}")
    print(f"  │ SMILES: {report.smiles}")
    print(f"  │ 剂量: {report.dose_mg:.0f}mg (治疗量{report.therapeutic_dose_mg:.0f}mg的{report.dose_ratio:.0f}倍)")
    print(f"  │ 等效: {report.dose_mg / 70:.0f}mg/kg")
    print(f"  │ ADMET: {report.admet_summary}")
    print(f"  │ 参数来源: PK={report.pk_params_source}, IC50={report.ic50_source}")
    print(f"  └{'─'*58}")

    # ── 时序过程表 ──
    print(f"\n  ┌─ 毒理反应时序过程 {'─'*42}")
    if not report.timeline:
        print(f"  │ (无数据)")
    else:
        # 表头
        print(f"  │ {'t(h)':>5} │ {'C(mg/L)':>8} │ {'5-HT':>5} │ {'HR':>4} │ {'BP':>4} │ {'T°C':>5} │ {'GCS':>3} │ {'Sz':>4} │ {'心损':>5} │ {'肝损':>5} │ {'神损':>5} │ {'阶段':>8}")
        print(f"  │ {'─'*5}─┼─{'─'*8}─┼─{'─'*5}─┼─{'─'*4}─┼─{'─'*4}─┼─{'─'*5}─┼─{'─'*3}─┼─{'─'*4}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*8}")

        # 采样: 最多40行，均匀采样
        n = len(report.timeline)
        step = max(1, n // 40)
        for i in range(0, n, step):
            s = report.timeline[i]
            print(f"  │ {s.time_h:5.1f} │ {s.conc_mg_per_l:8.4f} │ {s.serotonin_level:5.3f} │ "
                  f"{s.heart_rate:4.0f} │ {s.blood_pressure:4.0f} │ {s.body_temp:5.1f} │ "
                  f"{s.consciousness_gcs:3d} │ {s.seizure_activity:4.2f} │ "
                  f"{s.cardiac_damage:5.2f} │ {s.hepatic_damage:5.2f} │ {s.neuro_damage:5.2f} │ "
                  f"{s.syndrome_stage:>8}")
            if not s.alive:
                break

        # 确保最后一行
        if n > 1 and (n - 1) % step != 0:
            s = report.timeline[-1]
            print(f"  │ {s.time_h:5.1f} │ {s.conc_mg_per_l:8.4f} │ {s.serotonin_level:5.3f} │ "
                  f"{s.heart_rate:4.0f} │ {s.blood_pressure:4.0f} │ {s.body_temp:5.1f} │ "
                  f"{s.consciousness_gcs:3d} │ {s.seizure_activity:4.2f} │ "
                  f"{s.cardiac_damage:5.2f} │ {s.hepatic_damage:5.2f} │ {s.neuro_damage:5.2f} │ "
                  f"{s.syndrome_stage:>8}")

    print(f"  └{'─'*58}")

    # ── 硬件映射时序 ──
    print(f"\n  ┌─ 硬件指标时序 {'─'*46}")
    if report.timeline:
        print(f"  │ {'t(h)':>5} │ {'CPU%':>5} │ {'RAM%':>5} │ {'GPU°C':>5} │ {'可用RAM':>6} │ {'吞吐/s':>6} │ {'切换/s':>6}")
        print(f"  │ {'─'*5}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}")
        n = len(report.timeline)
        step = max(1, n // 40)
        for i in range(0, n, step):
            s = report.timeline[i]
            print(f"  │ {s.time_h:5.1f} │ {s.hw_cpu_pct:5.0f} │ {s.hw_ram_pct:5.0f} │ "
                  f"{s.hw_gpu_temp:5.0f} │ {s.hw_avail_ram_ratio:6.2f} │ {s.hw_event_throughput:6.0f} │ "
                  f"{s.hw_ctx_switch_rate:6.0f}")
            if not s.alive:
                break
        if n > 1 and (n - 1) % step != 0:
            s = report.timeline[-1]
            print(f"  │ {s.time_h:5.1f} │ {s.hw_cpu_pct:5.0f} │ {s.hw_ram_pct:5.0f} │ "
                  f"{s.hw_gpu_temp:5.0f} │ {s.hw_avail_ram_ratio:6.2f} │ {s.hw_event_throughput:6.0f} │ "
                  f"{s.hw_ctx_switch_rate:6.0f}")
    print(f"  └{'─'*58}")

    # ── 峰值指标 ──
    print(f"\n  ┌─ 峰值指标 {'─'*50}")
    print(f"  │ 峰值浓度: {report.peak_conc_mg_per_l:.4f} mg/L")
    print(f"  │ 峰值5-HT: {report.peak_5ht:.3f} (基线0.500)")
    print(f"  │ 最高体温: {report.max_temp_c:.1f}°C")
    print(f"  │ 最高心率: {report.max_hr_bpm:.0f}bpm")
    print(f"  │ 最高血压: {report.max_bp_mmhg:.0f}mmHg")
    print(f"  └{'─'*58}")

    # ── 死亡/存活 ──
    if report.time_of_death_h is not None:
        n_lethal = len([c for c in report.cause_chain if any(
            kw in c for kw in ["恶性高热", "心源性猝死", "脑功能衰竭", "失血性休克", "低氧血症", "癫痫持续状态", "多器官功能衰竭"]
        )])

        print(f"\n  ┌─ 死亡分析 {'─'*50}")
        print(f"  │ 死亡时间: {report.time_of_death_h:.1f}h (服药后)")
        print(f"  │ 主要死因: {report.cause_of_death}")
        print(f"  │")
        print(f"  │ 致死原因 (直接触发):")
        for i, cause in enumerate(report.cause_chain[:n_lethal], 1):
            print(f"  │   {i}. {cause}")
        if n_lethal < len(report.cause_chain):
            print(f"  │")
            print(f"  │ 促成因素 (加速死亡进程):")
            for i, cause in enumerate(report.cause_chain[n_lethal:], 1):
                print(f"  │   {i}. {cause}")
        print(f"  │")
        print(f"  │ 硬件崩溃路径:")
        n_hw = min(n_lethal, len(report.hardware_crash_path))
        for i, hw in enumerate(report.hardware_crash_path[:n_hw], 1):
            print(f"  │   {i}. {hw}")
        if n_hw < len(report.hardware_crash_path):
            print(f"  │")
            print(f"  │ 硬件降级路径:")
            for i, hw in enumerate(report.hardware_crash_path[n_hw:], 1):
                print(f"  │   {i}. {hw}")
        print(f"  └{'─'*58}")
    else:
        print(f"\n  ┌─ 存活结果 {'─'*50}")
        print(f"  │ {report.cause_of_death}")
        print(f"  └{'─'*58}")

    print(f"\n{'='*70}")
    print(f"  实验结束")
    print(f"{'='*70}\n")


def export_timeline(report: OverdoseReport) -> List[Dict]:
    """导出完整时序数据为字典列表，便于绘图/分析/对比。

    Returns:
        List of dicts, one per time step, with all vital + hardware fields.
    """
    rows = []
    for s in report.timeline:
        rows.append({
            "time_h": s.time_h,
            "conc_mg_per_l": s.conc_mg_per_l,
            "depot_mg": s.depot_mg,
            "peripheral_mg": s.peripheral_mg,
            "serotonin": s.serotonin_level,
            "inhibition": s.serotonin_reuptake_inhibition,
            "heart_rate": s.heart_rate,
            "blood_pressure": s.blood_pressure,
            "body_temp": s.body_temp,
            "o2_saturation": s.o2_saturation,
            "gcs": s.consciousness_gcs,
            "seizure": s.seizure_activity,
            "cardiac_damage": s.cardiac_damage,
            "hepatic_damage": s.hepatic_damage,
            "neuro_damage": s.neuro_damage,
            "hw_cpu_pct": s.hw_cpu_pct,
            "hw_ram_pct": s.hw_ram_pct,
            "hw_gpu_temp": s.hw_gpu_temp,
            "hw_avail_ram": s.hw_avail_ram_ratio,
            "hw_throughput": s.hw_event_throughput,
            "hw_ctx_switch": s.hw_ctx_switch_rate,
            "alive": s.alive,
            "stage": s.syndrome_stage,
        })
    return rows


# ══════════════════════════════════════════════════════════════
# 便捷入口
# ══════════════════════════════════════════════════════════════

def run_experiment(dose_mg: float = 4000) -> OverdoseReport:
    """运行舍曲林过量实验并打印报告"""
    report = run_sertraline_overdose(dose_mg=dose_mg, verbose=True)
    print_report(report)
    return report


__all__ = [
    "VitalSnapshot",
    "OverdoseReport",
    "run_sertraline_overdose",
    "run_experiment",
    "print_report",
    "export_timeline",
    "HARDWARE_MAP",
]