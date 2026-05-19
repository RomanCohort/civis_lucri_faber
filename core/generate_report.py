"""Generate figures and markdown report for sertraline overdose experiment."""

import sys
import importlib.util
import os

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

pkpd_mod = load_mod("drug_pipeline.pkpd", "core/drug_pipeline/pkpd.py")
admet_mod = load_mod("drug_pipeline.admet", "core/drug_pipeline/admet.py")
risk_mod = load_mod("drug_pipeline.risk_to_ic50", "core/drug_pipeline/risk_to_ic50.py")
reg_mod = load_mod("drug_pipeline.drug_registry", "core/drug_pipeline/drug_registry.py")

sys.modules["core"] = type(sys)("core")
sys.modules["core.drug_pipeline"] = type(sys)("core.drug_pipeline")
sys.modules["core.drug_pipeline.pkpd"] = pkpd_mod
sys.modules["core.drug_pipeline.admet"] = admet_mod
sys.modules["core.drug_pipeline.risk_to_ic50"] = risk_mod
sys.modules["core.drug_pipeline.drug_registry"] = reg_mod

exp = load_mod("sertraline_experiment", "core/sertraline_overdose_experiment.py")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

os.makedirs("docs/figures", exist_ok=True)

# ── Run experiments ──
doses = [200, 2000, 4000]
reports = {}
timelines = {}
for d in doses:
    r = exp.run_sertraline_overdose(dose_mg=d, body_weight_kg=70, simulation_hours=48, verbose=False)
    reports[d] = r
    timelines[d] = exp.export_timeline(r)

colors = {200: "#a6e3a1", 2000: "#f9e2af", 4000: "#f38ba8"}

# ── Figure 1: Concentration-Time Curves ──
fig, ax = plt.subplots(figsize=(10, 5))
for d in doses:
    tl = timelines[d]
    t = [row["time_h"] for row in tl]
    c = [row["conc_mg_per_l"] for row in tl]
    ax.plot(t, c, label=f"{d} mg", color=colors[d], linewidth=2)
ax.set_xlabel("Time (h)", fontsize=12)
ax.set_ylabel("Central Concentration (mg/L)", fontsize=12)
ax.set_title("Sertraline Plasma Concentration vs Time", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 48)
plt.tight_layout()
plt.savefig("docs/figures/concentration_time.png", dpi=150)
plt.close()

# ── Figure 2: 5-HT Syndrome Cascade ──
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]
for d in doses:
    tl = timelines[d]
    t = [row["time_h"] for row in tl]
    serotonin = [row["serotonin"] for row in tl]
    ax.plot(t, serotonin, label=f"{d} mg", color=colors[d], linewidth=2)
ax.set_ylabel("5-HT Level", fontsize=11)
ax.set_title("5-HT Syndrome Progression", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)
ax.set_xlim(0, 48)

ax = axes[0, 1]
for d in doses:
    tl = timelines[d]
    t = [row["time_h"] for row in tl]
    temp = [row["body_temp"] for row in tl]
    ax.plot(t, temp, label=f"{d} mg", color=colors[d], linewidth=2)
ax.axhspan(37.5, 38.5, alpha=0.1, color="yellow")
ax.axhspan(38.5, 40.0, alpha=0.1, color="orange")
ax.axhspan(40.0, 45.0, alpha=0.1, color="red")
ax.set_ylabel("Body Temperature (C)", fontsize=11)
ax.set_title("Thermoregulatory Response", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 48)

ax = axes[1, 0]
for d in doses:
    tl = timelines[d]
    t = [row["time_h"] for row in tl]
    gcs = [row["gcs"] for row in tl]
    ax.plot(t, gcs, label=f"{d} mg", color=colors[d], linewidth=2)
ax.set_xlabel("Time (h)", fontsize=11)
ax.set_ylabel("Glasgow Coma Scale", fontsize=11)
ax.set_title("Neurological Decline (GCS)", fontsize=12, fontweight="bold")
ax.set_ylim(0, 16)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=8, color="orange", linestyle="--", alpha=0.5)
ax.text(1, 8.5, "Coma threshold", fontsize=9, color="orange")
ax.set_xlim(0, 48)

ax = axes[1, 1]
for d in doses:
    tl = timelines[d]
    t = [row["time_h"] for row in tl]
    hr = [row["heart_rate"] for row in tl]
    ax.plot(t, hr, label=f"{d} mg", color=colors[d], linewidth=2)
ax.set_xlabel("Time (h)", fontsize=11)
ax.set_ylabel("Heart Rate (bpm)", fontsize=11)
ax.set_title("Cardiac Response", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
ax.text(1, 102, "Tachycardia", fontsize=9, color="gray")
ax.set_xlim(0, 48)

plt.suptitle("5-HT Syndrome Cascade: Sertraline Overdose", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("docs/figures/syndrome_cascade.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 3: Organ Damage Accumulation ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
organ_keys = ["cardiac_damage", "hepatic_damage", "neuro_damage"]
organ_labels = ["Cardiac Damage", "Hepatic Damage", "Neuro Damage"]

for i, (key, label) in enumerate(zip(organ_keys, organ_labels)):
    ax = axes[i]
    for d in doses:
        tl = timelines[d]
        t = [row["time_h"] for row in tl]
        dmg = [row[key] for row in tl]
        ax.plot(t, dmg, label=f"{d} mg", color=colors[d], linewidth=2)
    ax.set_xlabel("Time (h)", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 1.05)

plt.suptitle("Organ Damage Accumulation (Hill Equation)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("docs/figures/organ_damage.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 4: Hardware Mapping ──
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]
for d in doses:
    tl = timelines[d]
    t = [row["time_h"] for row in tl]
    cpu = [row["hw_cpu_pct"] for row in tl]
    ax.plot(t, cpu, label=f"{d} mg", color=colors[d], linewidth=2)
ax.axhline(y=90, color="red", linestyle="--", alpha=0.5)
ax.text(1, 91, "Critical", fontsize=9, color="red")
ax.set_ylabel("CPU Utilization (%)", fontsize=11)
ax.set_title("CPU% (Heart Rate Mapping)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 48)

ax = axes[0, 1]
for d in doses:
    tl = timelines[d]
    t = [row["time_h"] for row in tl]
    ram = [row["hw_ram_pct"] for row in tl]
    ax.plot(t, ram, label=f"{d} mg", color=colors[d], linewidth=2)
ax.axhline(y=85, color="red", linestyle="--", alpha=0.5)
ax.text(1, 86, "Critical", fontsize=9, color="red")
ax.set_ylabel("RAM Utilization (%)", fontsize=11)
ax.set_title("RAM% (Blood Pressure Mapping)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 48)

ax = axes[1, 0]
for d in doses:
    tl = timelines[d]
    t = [row["time_h"] for row in tl]
    gpu = [row["hw_gpu_temp"] for row in tl]
    ax.plot(t, gpu, label=f"{d} mg", color=colors[d], linewidth=2)
ax.axhline(y=90, color="red", linestyle="--", alpha=0.5)
ax.text(1, 91, "Thermal shutdown", fontsize=9, color="red")
ax.set_xlabel("Time (h)", fontsize=11)
ax.set_ylabel("GPU Temperature (C)", fontsize=11)
ax.set_title("GPU Temp (Body Temperature Mapping)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 48)

ax = axes[1, 1]
for d in doses:
    tl = timelines[d]
    t = [row["time_h"] for row in tl]
    throughput = [row["hw_throughput"] for row in tl]
    ax.plot(t, throughput, label=f"{d} mg", color=colors[d], linewidth=2)
ax.set_xlabel("Time (h)", fontsize=11)
ax.set_ylabel("Events/s", fontsize=11)
ax.set_title("Event Throughput (GCS Mapping)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 48)

plt.suptitle("Hardware-Biology Mapping: Sertraline Overdose", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("docs/figures/hardware_mapping.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 5: Dose-Response Summary ──
fig, ax = plt.subplots(figsize=(8, 5))
dose_range = [50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]
cmax_list = []
survival_list = []
death_time_list = []

for d in dose_range:
    r = exp.run_sertraline_overdose(dose_mg=d, body_weight_kg=70, simulation_hours=48, verbose=False)
    tl = exp.export_timeline(r)
    cmax = max(row["conc_mg_per_l"] for row in tl)
    cmax_list.append(cmax)
    survived = r.time_of_death_h is None
    survival_list.append(survived)
    death_time_list.append(r.time_of_death_h if not survived else 48.0)

ax2 = ax.twinx()
bar_colors = ["#a6e3a1" if s else "#f38ba8" for s in survival_list]
ax.bar(range(len(dose_range)), cmax_list, color=bar_colors, alpha=0.7)
ax2.plot(range(len(dose_range)), death_time_list, "ko-", markersize=6, label="Time to death (h)")
ax.set_xticks(range(len(dose_range)))
ax.set_xticklabels([str(d) for d in dose_range], fontsize=9)
ax.set_xlabel("Dose (mg)", fontsize=12)
ax.set_ylabel("Cmax (mg/L)", fontsize=12, color="blue")
ax2.set_ylabel("Time to Death (h)", fontsize=12, color="black")
ax.set_title("Dose-Response: Cmax and Survival", fontsize=14, fontweight="bold")
ax2.set_ylim(0, 52)
legend_elements = [
    Patch(facecolor="#a6e3a1", label="Survived"),
    Patch(facecolor="#f38ba8", label="Fatal"),
    plt.Line2D([0], [0], color="black", marker="o", label="Time to death"),
]
ax.legend(handles=legend_elements, fontsize=10, loc="upper left")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("docs/figures/dose_response.png", dpi=150)
plt.close()

# ── Figure 6: Vd Sensitivity ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
vd_values = [100, 200, 500, 1000, 1400, 1648, 2000, 3000]
cmax_100 = []
cardiac_100 = []
for vd in vd_values:
    r = exp.run_sertraline_overdose(dose_mg=100, body_weight_kg=70, simulation_hours=48, verbose=False,
                                     pk_overrides={"Vd": vd})
    tl = exp.export_timeline(r)
    cmax_100.append(max(row["conc_mg_per_l"] for row in tl))
    cardiac_100.append(max(row["cardiac_damage"] for row in tl))

ax = axes[0]
ax.plot(vd_values, cmax_100, "b-o", markersize=6, linewidth=2, label="Cmax")
ax.axvline(x=1648, color="green", linestyle="--", alpha=0.7, label="ADMET-inferred Vd")
ax.axvline(x=1400, color="purple", linestyle=":", alpha=0.7, label="Literature Vd")
ax.set_xlabel("Volume of Distribution Vd (L)", fontsize=11)
ax.set_ylabel("Cmax at 100mg (mg/L)", fontsize=11)
ax.set_title("Vd Sensitivity (100mg Therapeutic)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(vd_values, [d * 100 for d in cardiac_100], "r-o", markersize=6, linewidth=2, label="Max cardiac damage")
ax.axvline(x=1648, color="green", linestyle="--", alpha=0.7, label="ADMET-inferred Vd")
ax.axvline(x=1400, color="purple", linestyle=":", alpha=0.7, label="Literature Vd")
ax.set_xlabel("Volume of Distribution Vd (L)", fontsize=11)
ax.set_ylabel("Max Cardiac Damage (%)", fontsize=11)
ax.set_title("Vd Impact on Toxicity (100mg)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle("ADMET Parameter Sensitivity Analysis", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("docs/figures/sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()

print("All figures saved to docs/figures/")