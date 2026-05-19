"""ADMET risk → organ IC50 threshold bridge.

Converts Confluencia ADMET risk scores (0-1) into organ-specific IC50
concentration thresholds (mg/L) that Simulacrum's Hill-equation damage accumulation
can use. This replaces hardcoded IC50 values with data-driven ones.

Key insight: higher ADMET risk → lower IC50 → more toxic at lower concentrations.

Baseline IC50 values are derived from literature for "medium risk" compounds
(risk=0.5). The actual IC50 for a specific drug is scaled inversely by its
ADMET risk score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════
# Baseline IC50 values (mg/L) for "medium risk" (risk=0.5) compounds
# These represent typical toxicity thresholds from pharmacological literature.
# ══════════════════════════════════════════════════════════════

_BASELINE_IC50_BY_CLASS = {
    "SSRI": {
        "cardiac": 0.35,    # hERG blockade — SSRIs known for QT prolongation
        "hepatic": 0.60,    # CYP450 inhibition → hepatotoxicity
        "neuro": 0.40,      # 5-HT toxicity → excitotoxicity
        "hill": 2.0,        # steep dose-tox curve
    },
    "stimulant": {
        "cardiac": 0.25,    # sympathomimetic → cardiac strain
        "hepatic": 0.80,    # relatively less hepatic
        "neuro": 0.30,      # DA toxicity → neurodegeneration
        "hill": 1.8,
    },
    "sedative": {
        "cardiac": 0.50,    # respiratory depression → cardiac
        "hepatic": 0.70,    # CYP metabolism
        "neuro": 0.20,      # strong CNS depression
        "hill": 1.5,
    },
    "default": {
        "cardiac": 0.40,
        "hepatic": 0.60,
        "neuro": 0.35,
        "hill": 2.0,
    },
}


@dataclass
class OrganIC50Profile:
    """Organ-specific IC50 thresholds derived from ADMET predictions."""
    cardiac_ic50: float       # mg/L — hERG/QT prolongation threshold
    hepatic_ic50: float       # mg/L — hepatotoxicity threshold
    neuro_ic50: float         # mg/L — neurotoxicity threshold
    hill_coefficient: float   # steepness of dose-tox curve
    bbb_penetration: float    # 0-1 — how much drug reaches CNS
    damage_rate_cardiac: float  # damage accumulation rate per step
    damage_rate_hepatic: float
    damage_rate_neuro: float


def risk_to_ic50(risk_score: float, baseline_ic50: float) -> float:
    """Convert ADMET risk score to effective IC50.

    Higher risk → lower IC50 (more toxic at lower concentrations).
    Risk=0.5 → IC50 = baseline (unchanged).
    Risk=0.0 → IC50 = baseline × 10 (very safe, needs huge dose).
    Risk=1.0 → IC50 = baseline × 0.35 (very toxic, low threshold).

    Uses inverse-power scaling: scale = (0.5 / risk)^1.5
    """
    if risk_score < 0.01:
        return baseline_ic50 * 10.0
    scale = (0.5 / risk_score) ** 1.5
    return baseline_ic50 * max(scale, 0.01)


def admet_to_ic50(
    admet_result,
    drug_class: str = "default",
) -> OrganIC50Profile:
    """Convert ADMET prediction result to organ IC50 profile.

    Args:
        admet_result: ADMETResult from predict_admet()
        drug_class: Drug class key for baseline IC50 lookup

    Returns:
        OrganIC50Profile with drug-specific IC50 thresholds
    """
    baselines = _BASELINE_IC50_BY_CLASS.get(drug_class, _BASELINE_IC50_BY_CLASS["default"])

    # hERG risk → cardiac IC50
    cardiac_ic50 = risk_to_ic50(admet_result.hERG_risk, baselines["cardiac"])

    # Hepatotoxicity risk → hepatic IC50
    hepatic_ic50 = risk_to_ic50(admet_result.hepatotoxicity_risk, baselines["hepatic"])

    # Neuro toxicity: BBB penetration + hERG risk composite
    # BBB+ means drug reaches CNS → lower neuro IC50
    neuro_risk = 0.6 * admet_result.BBB_positive + 0.4 * admet_result.hERG_risk
    neuro_ic50 = risk_to_ic50(neuro_risk, baselines["neuro"])

    # Damage accumulation rates: higher risk → faster damage
    # Base rates from sertraline experiment: cardiac=0.005, hepatic=0.003, neuro=0.004
    cardiac_rate = 0.005 * (1.0 + admet_result.hERG_risk)
    hepatic_rate = 0.003 * (1.0 + admet_result.hepatotoxicity_risk)
    neuro_rate = 0.004 * (1.0 + neuro_risk)

    return OrganIC50Profile(
        cardiac_ic50=cardiac_ic50,
        hepatic_ic50=hepatic_ic50,
        neuro_ic50=neuro_ic50,
        hill_coefficient=baselines["hill"],
        bbb_penetration=admet_result.BBB_positive,
        damage_rate_cardiac=cardiac_rate,
        damage_rate_hepatic=hepatic_rate,
        damage_rate_neuro=neuro_rate,
    )


def smiles_to_pkpd_params(
    smiles: str,
    dose_mg: float,
    admet_result,
    body_weight_kg: float = 70.0,
) -> "PKPDParams":
    """Derive PK/PD parameters from SMILES-based ADMET for small molecules.

    This is the small-molecule adaptation of Confluencia's infer_pkpd_params().
    The original function was designed for circRNA therapeutics (binding/immune/
    inflammation inputs). This version uses ADMET-derived molecular properties
    to estimate PK parameters appropriate for conventional drugs.

    When RDKit is available, uses computed logP directly for Vd estimation,
    which is critical for lipophilic drugs (e.g. sertraline logP≈5.1 → Vd≈20 L/kg).
    Falls back to solubility proxy when RDKit is unavailable.

    Args:
        smiles: SMILES string
        dose_mg: Dose in mg
        admet_result: ADMETResult from predict_admet()
        body_weight_kg: Body weight for Vd scaling

    Returns:
        PKPDParams suitable for simulate_pkpd()
    """
    # Absorption rate: higher Caco-2 permeability → faster absorption
    # Base 0.5/h (oral, moderate absorption) + permeability boost
    caco2 = admet_result.caco2_permeability  # log Papp, range ~-8 to 2
    caco2_norm = np.clip((caco2 + 8.0) / 10.0, 0.0, 1.0)  # normalize to 0-1
    ka = float(np.clip(0.5 + 0.8 * caco2_norm, 0.05, 2.0))

    # Distribution: central → peripheral
    # Higher lipophilicity (lower solubility) → more tissue distribution
    sol_norm = np.clip((admet_result.aqueous_solubility + 2.0) / 4.0, 0.0, 1.0)
    k12 = float(np.clip(0.10 + 0.15 * (1.0 - sol_norm), 0.03, 0.60))

    # Return: peripheral → central
    k21 = float(np.clip(0.06 + 0.10 * sol_norm, 0.03, 0.60))

    # Elimination: CYP inhibition slows elimination
    # Higher CYP risk → slower elimination (drug inhibits its own metabolism)
    cyp_risk = admet_result.CYP_total_risk
    ke = float(np.clip(0.04 * (1.0 - 0.5 * cyp_risk), 0.01, 0.50))

    # Central compartment volume (L)
    # ── RDKit path: use computed logP directly ──
    # Vd scaling from logP (Lombardo et al. 2002, Pharm Res 19:201-208):
    #   log(Vd_ss) ≈ 0.37 × logP + 0.83  (steady-state, L/kg)
    # This gives much better estimates for lipophilic drugs:
    #   logP=2.5 → Vd≈3.2 L/kg, logP=5.1 → Vd≈20.5 L/kg (matches sertraline)
    computed_logp = None
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            from rdkit.Chem import Descriptors
            computed_logp = Descriptors.MolLogP(mol)
    except Exception:
        pass

    if computed_logp is not None:
        # Sigmoid Vd-logP model calibrated to literature values:
        #   logP=0 → ~0.7 L/kg (hydrophilic, plasma water)
        #   logP=3 → ~9 L/kg (moderate tissue distribution)
        #   logP=5 → ~22 L/kg (extensive tissue binding, matches sertraline 20 L/kg)
        #   logP>6 → ~24 L/kg (saturated)
        # Sigmoid: Vd = Vd_min + (Vd_max - Vd_min) / (1 + exp(-k*(logP - logP50)))
        Vd_min, Vd_max, logP50, k = 0.3, 25.0, 3.5, 1.2
        vd_per_kg = float(Vd_min + (Vd_max - Vd_min) / (1.0 + np.exp(-k * (computed_logp - logP50))))
    else:
        # Fallback: solubility proxy (less accurate for lipophilic drugs)
        vd_per_kg = float(np.clip(2.0 + 18.0 * (1.0 - sol_norm), 2.0, 25.0))

    v1_l = vd_per_kg * body_weight_kg  # total volume in L

    # PD parameters: Emax and EC50
    # Emax: maximum therapeutic effect (0-1 scale for neurotransmitter modulation)
    emax = float(np.clip(0.6 + 0.4 * admet_result.druglikeness_score, 0.2, 1.0))

    # EC50: concentration for 50% effect
    # Better druglikeness → lower EC50 (more potent)
    ec50 = float(np.clip(0.02 + 0.10 * (1.0 - admet_result.druglikeness_score), 0.005, 0.5))

    # Hill coefficient: positive cooperativity for receptor binding
    hill = float(np.clip(1.0 + 0.8 * cyp_risk, 0.8, 2.5))

    from .pkpd import PKPDParams
    return PKPDParams(
        ka=ka,
        k12=k12,
        k21=k21,
        ke=ke,
        v1_l=v1_l,
        emax=emax,
        ec50_mg_per_l=ec50,
        hill=hill,
    )


__all__ = [
    "OrganIC50Profile",
    "risk_to_ic50",
    "admet_to_ic50",
    "smiles_to_pkpd_params",
]