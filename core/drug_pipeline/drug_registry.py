"""Drug registry — SMILES-based definitions for known drugs.

Each entry provides the molecular identity (SMILES), pharmacological class,
therapeutic dosing, and neurotransmitter targets. This enables the Confluencia
pipeline to compute ADMET/PK/PD parameters from molecular structure rather
than hardcoding them.

SMILES sources: PubChem canonical SMILES
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DrugDefinition:
    """Complete definition of a known drug for the pipeline."""
    name: str                          # Common name + brand
    smiles: str                        # Canonical SMILES (PubChem)
    drug_class: str                    # Pharmacological class key
    therapeutic_dose_mg: float         # Typical daily dose (mg)
    freq_per_day: float                # Dosing frequency
    target_neurotransmitters: dict[str, str]  # {"serotonin": "increase", ...}
    therapeutic_ed50_mgkg: float       # For dose-tox TI calculation
    notes: str = ""                    # Clinical notes
    cyp_substrate: dict[str, float] = field(default_factory=dict)
    #   Maps CYP isoform → fractional contribution to clearance (fm)
    #   e.g. {"CYP2C19": 0.40, "CYP3A4": 0.30}
    cyp_inhibitor_class: dict[str, str] = field(default_factory=dict)
    #   Maps CYP isoform → Flockhart classification
    #   "strong" / "moderate" / "weak"
    receptor_targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    #   {"5-HT1A": {"effect": "partial_agonist", "ki_nm": 21.0}, ...}
    #   When populated, enables receptor-level PD computation
    addiction_risk: str = "low"  # "low"/"moderate"/"high"/"very_high"


DRUG_REGISTRY: dict[str, DrugDefinition] = {
    "sertraline": DrugDefinition(
        name="Sertraline (Zoloft)",
        smiles="CNCc1ccc(Cl)cc1N2C3=CC=CC=C3C4=C2C=C(C=C4)Cl",
        drug_class="SSRI",
        therapeutic_dose_mg=100.0,
        freq_per_day=1.0,
        target_neurotransmitters={
            "serotonin": "increase",    # SERT inhibition → 5-HT↑
            "dopamine": "slight_increase",  # mild DA effect at high dose
        },
        therapeutic_ed50_mgkg=2.0,      # ~2 mg/kg effective
        notes="SSRI; CYP2D6/3A4 inhibitor; hERG risk at high dose; t1/2≈26h",
        cyp_substrate={"CYP2C19": 0.40, "CYP3A4": 0.30, "CYP2D6": 0.10},
        cyp_inhibitor_class={"CYP2D6": "moderate", "CYP3A4": "weak"},
        receptor_targets={
            "SERT": {"effect": "antagonist", "ki_nm": 0.14, "intrinsic_activity": 0.0},
            "5-HT2A": {"effect": "antagonist", "ki_nm": 325.0, "intrinsic_activity": 0.0},
            "DAT": {"effect": "antagonist", "ki_nm": 25.0, "intrinsic_activity": 0.0},
        },
        addiction_risk="low",
    ),
    "fluoxetine": DrugDefinition(
        name="Fluoxetine (Prozac)",
        smiles="CN(C)CCC1=Cc2ccc(OC)cc2C1=O",
        drug_class="SSRI",
        therapeutic_dose_mg=20.0,
        freq_per_day=1.0,
        target_neurotransmitters={
            "serotonin": "increase",
        },
        therapeutic_ed50_mgkg=1.0,
        notes="SSRI; long half-life (~4-6 days); strong CYP2D6 inhibitor",
        cyp_substrate={"CYP2D6": 0.50, "CYP2C19": 0.20, "CYP3A4": 0.10},
        cyp_inhibitor_class={"CYP2D6": "strong", "CYP3A4": "moderate", "CYP2C19": "moderate"},
        receptor_targets={
            "SERT": {"effect": "antagonist", "ki_nm": 1.0, "intrinsic_activity": 0.0},
            "5-HT2C": {"effect": "antagonist", "ki_nm": 8.0, "intrinsic_activity": 0.0},
            "NET": {"effect": "antagonist", "ki_nm": 1000.0, "intrinsic_activity": 0.0},
        },
        addiction_risk="low",
    ),
    "paroxetine": DrugDefinition(
        name="Paroxetine (Paxil)",
        smiles="c1cc2c(cc1OC)COC(=O)C2c3ccc(F)cc3NC",
        drug_class="SSRI",
        therapeutic_dose_mg=20.0,
        freq_per_day=1.0,
        target_neurotransmitters={
            "serotonin": "increase",
        },
        therapeutic_ed50_mgkg=0.5,
        notes="SSRI; most potent SERT inhibitor; strong CYP2D6 inhibitor",
        cyp_substrate={"CYP2D6": 0.70, "CYP3A4": 0.15, "CYP2C19": 0.10},
        cyp_inhibitor_class={"CYP2D6": "strong", "CYP2C19": "moderate"},
        receptor_targets={
            "SERT": {"effect": "antagonist", "ki_nm": 0.06, "intrinsic_activity": 0.0},
            "NET": {"effect": "antagonist", "ki_nm": 45.0, "intrinsic_activity": 0.0},
        },
        addiction_risk="low",
    ),
    "amphetamine": DrugDefinition(
        name="Amphetamine",
        smiles="CC(N)Cc1ccccc1",
        drug_class="stimulant",
        therapeutic_dose_mg=10.0,
        freq_per_day=1.0,
        target_neurotransmitters={
            "dopamine": "increase",
            "norepinephrine": "increase",
        },
        therapeutic_ed50_mgkg=0.5,
        notes="Stimulant; DA/NE release + reuptake inhibition; high abuse potential",
        cyp_substrate={"CYP2D6": 0.60, "CYP3A4": 0.20},
        receptor_targets={
            "DAT": {"effect": "antagonist", "ki_nm": 0.18, "intrinsic_activity": 0.0},  # reuptake inhibitor
            "NET": {"effect": "antagonist", "ki_nm": 0.35, "intrinsic_activity": 0.0},
            "5-HT2A": {"effect": "agonist", "ki_nm": 500.0, "intrinsic_activity": 0.3},
        },
        addiction_risk="high",
    ),
    "diazepam": DrugDefinition(
        name="Diazepam (Valium)",
        smiles="C1=CC=C2C(=C1)C3=CC=CC=C3N2C(=O)CN4C(=O)C5=CC=CC=C5C(=O)N4C",
        drug_class="sedative",
        therapeutic_dose_mg=5.0,
        freq_per_day=2.0,
        target_neurotransmitters={
            "gaba": "increase",         # GABA-A positive allosteric modulator
        },
        therapeutic_ed50_mgkg=0.5,
        notes="Benzodiazepine; GABA-A modulator; t1/2≈20-50h; CYP3A4/2C19 metabolized",
        cyp_substrate={"CYP3A4": 0.60, "CYP2C19": 0.30, "CYP2C9": 0.05},
        receptor_targets={
            "GABA-A": {"effect": "pam", "ki_nm": 0.05, "intrinsic_activity": 0.0},
        },
        addiction_risk="moderate",
    ),
    "haloperidol": DrugDefinition(
        name="Haloperidol (Haldol)",
        smiles="C1=CC=C2C(=C1)C3=CC=CC=C3N2CCC4CC(=O)NC4=O",
        drug_class="antipsychotic",
        therapeutic_dose_mg=5.0,
        freq_per_day=2.0,
        target_neurotransmitters={
            "dopamine": "decrease",     # D2 antagonist
        },
        therapeutic_ed50_mgkg=0.5,
        notes="Typical antipsychotic; D2 antagonist; high hERG risk; EPS side effects",
        cyp_substrate={"CYP2D6": 0.50, "CYP3A4": 0.30, "CYP1A2": 0.10},
        cyp_inhibitor_class={"CYP2D6": "moderate", "CYP3A4": "weak"},
        receptor_targets={
            "D2": {"effect": "antagonist", "ki_nm": 0.5, "intrinsic_activity": 0.0},
            "5-HT2A": {"effect": "antagonist", "ki_nm": 30.0, "intrinsic_activity": 0.0},
            "alpha1": {"effect": "antagonist", "ki_nm": 10.0, "intrinsic_activity": 0.0},
        },
        addiction_risk="low",
    ),
    "lithium": DrugDefinition(
        name="Lithium carbonate",
        smiles="[Li+].[Li+].[O-]C([O-])=O",  # ionic, not typical SMILES
        drug_class="mood_stabilizer",
        therapeutic_dose_mg=300.0,
        freq_per_day=2.0,
        target_neurotransmitters={
            "serotonin": "slight_increase",
            "dopamine": "moderate",
            "gaba": "increase",
        },
        therapeutic_ed50_mgkg=10.0,
        notes="Mood stabilizer; narrow TI; renal/hepatic toxicity at overdose",
        cyp_substrate={},  # lithium几乎完全经肾排泄
        receptor_targets={},  # acts via GSK-3β/inositol, not classical receptors
        addiction_risk="low",
    ),
    "ketamine": DrugDefinition(
        name="Ketamine",
        smiles="CCC1=CC(=O)NC2=C1C=CC=C2Cl",
        drug_class="hallucinogen",
        therapeutic_dose_mg=50.0,  # IV anesthetic dose
        freq_per_day=0.5,          # single IV bolus
        target_neurotransmitters={
            "glutamate": "decrease",   # NMDA antagonist
            "dopamine": "increase",
        },
        therapeutic_ed50_mgkg=2.0,
        notes="NMDA antagonist; rapid antidepressant at sub-anesthetic dose; dissociative",
        cyp_substrate={"CYP3A4": 0.50, "CYP2C9": 0.10, "CYP2B6": 0.10},
        receptor_targets={
            "NMDA": {"effect": "antagonist", "ki_nm": 0.5, "intrinsic_activity": 0.0},
            "AMPA": {"effect": "agonist", "ki_nm": 5000.0, "intrinsic_activity": 0.2},
        },
        addiction_risk="moderate",
    ),
    "morphine": DrugDefinition(
        name="Morphine",
        smiles="C1=CC=C2C(C3=CC=CC=C3N2C4CC5C6C(O)C7C(OC)C=CC(C7)C6C5C4O)=C1O",
        drug_class="opioid",
        therapeutic_dose_mg=10.0,
        freq_per_day=4.0,
        target_neurotransmitters={
            "dopamine": "increase",    # μ-opioid → DA release in VTA
            "gaba": "decrease",        # disinhibition of DA neurons
        },
        therapeutic_ed50_mgkg=1.0,
        notes="μ-opioid agonist; respiratory depression risk; high abuse potential",
        cyp_substrate={"CYP3A4": 0.40, "CYP2D6": 0.30},
        receptor_targets={
            "mu-opioid": {"effect": "agonist", "ki_nm": 1.8, "intrinsic_activity": 1.0},
            "delta": {"effect": "agonist", "ki_nm": 200.0, "intrinsic_activity": 0.5},
            "kappa": {"effect": "agonist", "ki_nm": 50.0, "intrinsic_activity": 0.3},
        },
        addiction_risk="very_high",
    ),
    "buspirone": DrugDefinition(
        name="Buspirone (Buspar)",
        smiles="CC1=CC=C(C=C1)N2CCN(CC2)CCCC(=O)N3CCN(CC3)C",
        drug_class="anxiolytic",
        therapeutic_dose_mg=15.0,
        freq_per_day=2.0,
        target_neurotransmitters={
            "serotonin": "modulate",      # 5-HT1A partial agonist
            "dopamine": "slight_increase", # D2 antagonist at high dose
        },
        therapeutic_ed50_mgkg=1.0,
        notes="5-HT1A partial agonist; delayed anxiolytic (2-4 weeks); no sedation/dependence",
        cyp_substrate={"CYP3A4": 0.60, "CYP2D6": 0.10},
        cyp_inhibitor_class={"CYP3A4": "weak"},
        receptor_targets={
            "5-HT1A": {"effect": "partial_agonist", "ki_nm": 21.0, "intrinsic_activity": 0.5},
            "D2": {"effect": "antagonist", "ki_nm": 100.0, "intrinsic_activity": 0.0},
        },
        addiction_risk="low",
    ),
    "buprenorphine": DrugDefinition(
        name="Buprenorphine (Suboxone)",
        smiles="CC1(C)C2C(CC1)C3C4C5CC6C(C5)C4C3N2C(=O)CC7=CC=CC=C67O",
        drug_class="opioid",
        therapeutic_dose_mg=8.0,  # sublingual
        freq_per_day=1.0,
        target_neurotransmitters={
            "dopamine": "increase",     # mu-opioid partial agonist → DA release
            "gaba": "decrease",         # disinhibition
        },
        therapeutic_ed50_mgkg=0.05,
        notes="mu-opioid partial agonist; kappa antagonist; ceiling effect; MAT for opioid dependence",
        cyp_substrate={"CYP3A4": 0.70, "CYP2C8": 0.20},
        cyp_inhibitor_class={"CYP3A4": "moderate", "CYP2D6": "weak"},
        receptor_targets={
            "mu-opioid": {"effect": "partial_agonist", "ki_nm": 0.8, "intrinsic_activity": 0.4},
            "kappa": {"effect": "antagonist", "ki_nm": 1.5, "intrinsic_activity": 0.0},
            "delta": {"effect": "antagonist", "ki_nm": 150.0, "intrinsic_activity": 0.0},
        },
        addiction_risk="moderate",
    ),
    "propranolol": DrugDefinition(
        name="Propranolol (Inderal)",
        smiles="CC(C)NCC(COc1cccc2ccccc12)O",
        drug_class="beta_blocker",
        therapeutic_dose_mg=40.0,
        freq_per_day=2.0,
        target_neurotransmitters={
            "norepinephrine": "decrease",  # beta-adrenergic blockade
        },
        therapeutic_ed50_mgkg=0.5,
        notes="Non-selective beta-blocker; crosses BBB; anxiolytic for performance anxiety; prevents peripheral panic symptoms",
        cyp_substrate={"CYP2D6": 0.50, "CYP1A2": 0.30, "CYP3A4": 0.10},
        cyp_inhibitor_class={"CYP2D6": "moderate"},
        receptor_targets={
            "beta1": {"effect": "antagonist", "ki_nm": 1.5, "intrinsic_activity": 0.0},
            "beta2": {"effect": "antagonist", "ki_nm": 1.0, "intrinsic_activity": 0.0},
            "5-HT1A": {"effect": "antagonist", "ki_nm": 10.0, "intrinsic_activity": 0.0},
        },
        addiction_risk="low",
    ),
    "suvorexant": DrugDefinition(
        name="Suvorexant (Belsomra)",
        smiles="CC1=CC=C(C=C1)C2=NC(=NN2C3=CC=CC=C3Cl)C(=O)OC(C)(C)C",
        drug_class="dora",
        therapeutic_dose_mg=10.0,
        freq_per_day=1.0,
        target_neurotransmitters={
            "norepinephrine": "decrease",  # reduces orexin-driven arousal
            "serotonin": "slight_increase", # indirect via reduced arousal
        },
        therapeutic_ed50_mgkg=0.5,
        notes="Dual orexin receptor antagonist (DORA); promotes sleep onset/maintenance; no dependence",
        cyp_substrate={"CYP3A4": 0.80, "CYP2C19": 0.10},
        cyp_inhibitor_class={},
        receptor_targets={
            "orexin1": {"effect": "antagonist", "ki_nm": 5.0, "intrinsic_activity": 0.0},
            "orexin2": {"effect": "antagonist", "ki_nm": 5.0, "intrinsic_activity": 0.0},
        },
        addiction_risk="low",
    ),
    "minocycline": DrugDefinition(
        name="Minocycline",
        smiles="CC1C2C(C3C(C(=O)C(=C(C3O)C(=O)C2=C(C4=C1C=CC(=C4O)N(C)C)O)O)N(C)C)O",
        drug_class="antibiotic",
        therapeutic_dose_mg=100.0,
        freq_per_day=2.0,
        target_neurotransmitters={},  # no direct NT effects
        therapeutic_ed50_mgkg=5.0,
        notes="Tetracycline; crosses BBB; microglial inhibitor (TLR2/4); anti-inflammatory; neuroprotective adjunct",
        cyp_substrate={},  # minimal CYP metabolism
        cyp_inhibitor_class={"CYP3A4": "weak"},
        receptor_targets={},  # acts via TLR, not classical receptors
        addiction_risk="low",
    ),
    "clonidine": DrugDefinition(
        name="Clonidine (Catapres)",
        smiles="IC1=CC(=C(C=C1Cl)Cl)NC=C2C=CCN2",
        drug_class="alpha2_agonist",
        therapeutic_dose_mg=0.1,
        freq_per_day=2.0,
        target_neurotransmitters={
            "norepinephrine": "decrease",  # alpha2 autoreceptor activation
        },
        therapeutic_ed50_mgkg=0.01,
        notes="Alpha-2 adrenergic agonist; reduces sympathetic outflow; anxiolytic; opioid withdrawal adjunct",
        cyp_substrate={"CYP2D6": 0.50, "CYP3A4": 0.20},
        cyp_inhibitor_class={},
        receptor_targets={
            "alpha2": {"effect": "agonist", "ki_nm": 2.0, "intrinsic_activity": 1.0},
        },
        addiction_risk="low",
    ),
}


def get_drug(name: str) -> DrugDefinition | None:
    """Look up a drug by name (case-insensitive)."""
    return DRUG_REGISTRY.get(name.lower())


def list_drugs() -> list[str]:
    """List all registered drug names."""
    return sorted(DRUG_REGISTRY.keys())


def register_drug(drug: DrugDefinition) -> None:
    """Register a new drug definition."""
    key = drug.name.split()[0].lower()  # use first word as key
    DRUG_REGISTRY[key] = drug


__all__ = [
    "DrugDefinition",
    "DRUG_REGISTRY",
    "get_drug",
    "list_drugs",
    "register_drug",
]
