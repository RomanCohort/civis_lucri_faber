"""Drug pipeline sub-package — Confluencia PK/PD/ADMET/Toxicity modules.

Provides SMILES-driven pharmacological parameter inference:
  - pkpd: Three-compartment PK model + sigmoid Emax PD
  - admet: Multi-endpoint ADMET prediction (hERG, AMES, CYP450, BBB, etc.)
  - dose_tox: Dose-dependent toxicity estimation with organ-specific curves
  - toxicophore: Structural alert detection (PAINS, Brenk, etc.)
  - risk_to_ic50: Bridge layer — ADMET risk scores → organ IC50 thresholds
  - drug_registry: Known drug definitions (SMILES, class, dose, targets)
  - receptor_pd: Receptor-subtype pharmacodynamics (5-HT1A, D2, GABA-A, etc.)
"""

from .pkpd import PKPDParams, simulate_pkpd, summarize_pkpd_curve
from .admet import ADMETResult, ADMETPredictor, predict_admet
from .dose_tox import (
    DoseToxicityModel, DoseToxicityReport, OrganToxicity,
    estimate_dose_toxicity,
)
from .toxicophore import (
    ToxicophoreReport, ToxicophoreMatch, ToxicophoreDetector,
    detect_toxicophores,
)
from .risk_to_ic50 import OrganIC50Profile, admet_to_ic50, smiles_to_pkpd_params
from .drug_registry import DrugDefinition, DRUG_REGISTRY, get_drug, list_drugs, register_drug
from .ddi import (
    DDIPairRecord, DDIResult, assess_ddi,
    combine_pd_deltas, compute_step_ke_modifiers,
)
from .receptor_pd import (
    ReceptorSubtype, ReceptorPDTarget,
    RECEPTOR_REGISTRY, DRUG_RECEPTOR_AFFINITY,
    build_receptor_pd_targets, compute_receptor_deltas,
    aggregate_receptor_to_nt, get_receptor_time_profile,
)

__all__ = [
    # PK/PD
    "PKPDParams", "simulate_pkpd", "summarize_pkpd_curve",
    # ADMET
    "ADMETResult", "ADMETPredictor", "predict_admet",
    # Dose-tox
    "DoseToxicityModel", "DoseToxicityReport", "OrganToxicity",
    "estimate_dose_toxicity",
    # Toxicophore
    "ToxicophoreReport", "ToxicophoreMatch", "ToxicophoreDetector",
    "detect_toxicophores",
    # Bridge
    "OrganIC50Profile", "admet_to_ic50", "smiles_to_pkpd_params",
    # Registry
    "DrugDefinition", "DRUG_REGISTRY", "get_drug", "list_drugs", "register_drug",
    # DDI
    "DDIPairRecord", "DDIResult", "assess_ddi",
    "combine_pd_deltas", "compute_step_ke_modifiers",
    # Receptor PD
    "ReceptorSubtype", "ReceptorPDTarget",
    "RECEPTOR_REGISTRY", "DRUG_RECEPTOR_AFFINITY",
    "build_receptor_pd_targets", "compute_receptor_deltas",
    "aggregate_receptor_to_nt", "get_receptor_time_profile",
]