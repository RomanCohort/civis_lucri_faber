"""Simulacrum Neuro Monitor - 仿生大脑参数实时监测面板

基于 Confluencia Studio 架构，使用 PyQt6 + Matplotlib 显示所有大脑参数的
分类实时曲线。

运行:
    # 连接 agent (从 chat_main 使用)
    python monitor_panel.py --agent

    # 独立演示模式 (模拟数据)
    python monitor_panel.py

    # 指定刷新率
    python monitor_panel.py --interval 500
"""
import sys
import os
import time
import collections
import argparse
import math
import random

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QLabel, QScrollArea, QPushButton, QComboBox,
    QToolBar, QStatusBar, QGroupBox, QSplitter, QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ═══════════════════════════════════════════════════════════════
# Catppuccin Mocha 暗色主题
# ═══════════════════════════════════════════════════════════════
COLORS = {
    "base":      "#1e1e2e",
    "mantle":    "#181825",
    "crust":     "#11111b",
    "surface0":  "#313244",
    "surface1":  "#45475a",
    "surface2":  "#585b70",
    "overlay0":  "#6c7086",
    "subtext0":  "#a6adc8",
    "text":      "#cdd6f4",
    "lavender":  "#b4befe",
    "blue":      "#89b4fa",
    "sapphire":  "#74c7ec",
    "sky":       "#89dceb",
    "teal":      "#94e2d5",
    "green":     "#a6e3a1",
    "yellow":    "#f9e2af",
    "peach":     "#fab387",
    "maroon":    "#eba0ac",
    "red":       "#f38ba8",
    "mauve":     "#cba6f7",
    "pink":      "#f5c2e7",
    "flamingo":  "#f2cdcd",
    "rosewater": "#f5e0dc",
}

CHART_FG = COLORS["blue"]
CHART_BG = COLORS["mantle"]
CHART_GRID = COLORS["surface0"]
TEXT_COLOR = COLORS["text"]
SUBTEXT_COLOR = COLORS["subtext0"]


# ═══════════════════════════════════════════════════════════════
# 参数分类定义
# ═══════════════════════════════════════════════════════════════

CATEGORY_DEFS = {
    "Neurotransmitters": {
        "description": "神经递质系统",
        "params": {
            "nt_dopamine":        {"label": "Dopamine",   "color": COLORS["red"],     "range": (0, 1)},
            "nt_serotonin":       {"label": "Serotonin",  "color": COLORS["green"],   "range": (0, 1)},
            "nt_acetylcholine":   {"label": "ACh",        "color": COLORS["blue"],    "range": (0, 1)},
            "nt_norepinephrine":  {"label": "NE",         "color": COLORS["yellow"],  "range": (0, 1)},
            "nt_motivation":      {"label": "Motivation",  "color": COLORS["mauve"],   "range": (0, 1)},
            "nt_arousal":         {"label": "NT Arousal",  "color": COLORS["peach"],   "range": (0, 1)},
        },
    },
    "Hormones": {
        "description": "激素系统",
        "params": {
            "hormone_cortisol":     {"label": "Cortisol",      "color": COLORS["red"],    "range": (0, 1)},
            "hormone_adrenaline":   {"label": "Adrenaline",    "color": COLORS["peach"],  "range": (0, 1)},
            "hormone_melatonin":    {"label": "Melatonin",     "color": COLORS["mauve"],  "range": (0, 1)},
            "hormone_oxytocin":     {"label": "Oxytocin",      "color": COLORS["teal"],   "range": (0, 1)},
            "encoding_modulation":  {"label": "Encoding Mod",  "color": COLORS["sapphire"],"range": (0, 1.5)},
            "exploration_modulation":{"label": "Explore Mod",  "color": COLORS["green"],  "range": (0, 1.5)},
            "gr_sensitivity":       {"label": "GR Sensitivity", "color": COLORS["flamingo"],"range": (0.2, 1.1)},
            "mr_sensitivity":       {"label": "MR Sensitivity", "color": COLORS["rosewater"],"range": (0.4, 1.1)},
        },
    },
    "Brainstem": {
        "description": "脑干生命体征",
        "params": {
            "bsm_arousal":            {"label": "Arousal",        "color": COLORS["peach"],  "range": (0, 1)},
            "bsm_consciousness_gate": {"label": "Conscious Gate", "color": COLORS["yellow"], "range": (0, 1)},
            "bsm_cortical_activation":{"label": "Cortical Act.", "color": COLORS["blue"],   "range": (0, 1)},
            "bsm_respiratory_rate":   {"label": "Resp. Rate",     "color": COLORS["sky"],    "range": (0, 40)},
            "bsm_heart_rate":         {"label": "Heart Rate",     "color": COLORS["red"],    "range": (0, 200)},
            "bsm_blood_pressure":     {"label": "Blood Press.",   "color": COLORS["maroon"], "range": (0, 200)},
            "bsm_pain_gating":        {"label": "Pain Gate",      "color": COLORS["flamingo"],"range": (0, 1)},
        },
    },
    "Autonomic": {
        "description": "自主神经系统",
        "params": {
            "ans_sympathetic":    {"label": "Sympathetic",  "color": COLORS["red"],    "range": (0, 1)},
            "ans_parasympathetic":{"label": "Parasympath.", "color": COLORS["green"],  "range": (0, 1)},
            "ans_hrv":            {"label": "HRV",          "color": COLORS["blue"],   "range": (0, 1)},
            "heart_rate":         {"label": "Heart Rate",   "color": COLORS["maroon"], "range": (0, 200)},
            "blood_pressure":     {"label": "Blood Press.",  "color": COLORS["peach"],  "range": (0, 200)},
        },
    },
    "HPA & Stress": {
        "description": "HPA轴与应激",
        "params": {
            "cortisol_level": {"label": "Cortisol Lvl", "color": COLORS["red"],    "range": (0, 1)},
            "cortisol":       {"label": "Cortisol",     "color": COLORS["maroon"], "range": (0, 1)},
            "hpa_crh":        {"label": "CRH",          "color": COLORS["peach"],  "range": (0, 1)},
            "hpa_acth":       {"label": "ACTH",         "color": COLORS["yellow"], "range": (0, 1)},
        },
    },
    "Glial & Metabolic": {
        "description": "胶质与代谢",
        "params": {
            "brain_waste":            {"label": "Brain Waste",     "color": COLORS["red"],    "range": (0, 1)},
            "neuroinflammation":      {"label": "Neuroinflam.",    "color": COLORS["peach"],  "range": (0, 1)},
            "myelination_level":      {"label": "Myelination",     "color": COLORS["blue"],   "range": (0, 1)},
            "brain_health":           {"label": "Brain Health",    "color": COLORS["green"],  "range": (0, 1)},
            "glymphatic_clearance":   {"label": "Glymphatic",      "color": COLORS["sky"],    "range": (0, 1)},
            "allostatic_load":        {"label": "Allostatic Load", "color": COLORS["yellow"], "range": (0, 1)},
            "regulatory_capacity":    {"label": "Regul. Capacity", "color": COLORS["teal"],   "range": (0, 1)},
        },
    },
    "Limbic & Emotion": {
        "description": "边缘系统与情绪",
        "params": {
            "limbic_valence":            {"label": "Valence",     "color": COLORS["green"],  "range": (-1, 1)},
            "limbic_arousal":            {"label": "Limbic Ar.",  "color": COLORS["peach"],  "range": (0, 1)},
            "limbic_emotional_attention":{"label": "Emot. Attn.", "color": COLORS["yellow"], "range": (0, 1)},
            "mood_valence":              {"label": "Mood Val.",   "color": COLORS["teal"],   "range": (-1, 1)},
            "mood_arousal":              {"label": "Mood Ar.",    "color": COLORS["sky"],    "range": (0, 1)},
            "regulation_capacity":       {"label": "Regul. Cap.", "color": COLORS["blue"],   "range": (0, 1)},
        },
    },
    "Executive Control": {
        "description": "前额叶与基底节",
        "params": {
            "pfc_inhibition":    {"label": "PFC Inhibit.",  "color": COLORS["blue"],   "range": (0, 1)},
            "pfc_maturity":      {"label": "PFC Maturity",  "color": COLORS["sapphire"],"range": (0, 1)},
            "pfc_plan_depth":    {"label": "Plan Depth",    "color": COLORS["lavender"],"range": (0, 10)},
            "bg_td_error":       {"label": "BG TD Error",   "color": COLORS["red"],    "range": (-1, 1)},
            "bg_habit_strength": {"label": "Habit Str.",    "color": COLORS["peach"],  "range": (0, 1)},
            "conscious_load":    {"label": "Consc. Load",   "color": COLORS["yellow"], "range": (0, 1)},
        },
    },
    "Circadian & Sleep": {
        "description": "昼夜节律与睡眠",
        "params": {
            "scn_alertness":         {"label": "Alertness",       "color": COLORS["yellow"], "range": (0, 1)},
            "scn_melatonin":         {"label": "SCN Melatonin",   "color": COLORS["mauve"],  "range": (0, 1)},
            "scn_cortisol_rhythm":   {"label": "Cortisol Rhythm", "color": COLORS["red"],    "range": (0, 1)},
            "scn_wake_drive":        {"label": "Wake Drive",      "color": COLORS["peach"],  "range": (0, 1)},
            "scn_sleep_pressure":    {"label": "Sleep Pressure",  "color": COLORS["blue"],   "range": (0, 1)},
            "scn_temperature":       {"label": "Core Temp.",      "color": COLORS["maroon"], "range": (35, 40)},
            "sleep_fatigue":         {"label": "Fatigue",         "color": COLORS["surface1"],"range": (0, 1)},
        },
    },
    "Predictive Coding": {
        "description": "预测编码",
        "params": {
            "free_energy":           {"label": "Free Energy",     "color": COLORS["red"],    "range": (0, 2)},
            "prediction_error":      {"label": "Pred. Error",     "color": COLORS["peach"],  "range": (0, 2)},
            "precision":             {"label": "Precision",       "color": COLORS["blue"],   "range": (0, 2)},
            "active_inference_drive":{"label": "Active Infer.",   "color": COLORS["teal"],   "range": (0, 1)},
        },
    },
    "Social & Empathy": {
        "description": "社会认知与共情",
        "params": {
            "social_engagement": {"label": "Soc. Engage.",  "color": COLORS["green"],  "range": (0, 1)},
            "social_capacity":   {"label": "Soc. Capacity", "color": COLORS["teal"],   "range": (0, 1)},
            "mirror_resonance":  {"label": "Mirror Res.",   "color": COLORS["sky"],    "range": (0, 1)},
            "affective_empathy": {"label": "Affec. Emp.",   "color": COLORS["pink"],   "range": (0, 1)},
            "cognitive_empathy": {"label": "Cog. Emp.",     "color": COLORS["mauve"],  "range": (0, 1)},
            "compassion":        {"label": "Compassion",    "color": COLORS["flamingo"],"range": (0, 1)},
            "pain_resonance":    {"label": "Pain Res.",     "color": COLORS["red"],    "range": (0, 1)},
            "empathy_level":     {"label": "Empathy Lvl",   "color": COLORS["rosewater"],"range": (0, 1)},
        },
    },
    "Self & Plasticity": {
        "description": "自我意识与可塑性",
        "params": {
            "self_coherence":         {"label": "Self Coherence",  "color": COLORS["lavender"],"range": (0, 1)},
            "self_endorsement":       {"label": "Self Endorse.",   "color": COLORS["mauve"],  "range": (0, 1)},
            "narrative_continuity":   {"label": "Narrative Cont.", "color": COLORS["blue"],   "range": (0, 1)},
            "plasticity_bdnf":        {"label": "BDNF",            "color": COLORS["green"],  "range": (0, 1)},
            "plasticity_synapses":    {"label": "Synaptic Str.",   "color": COLORS["teal"],   "range": (0, 1)},
            "info_gain":              {"label": "Info Gain",       "color": COLORS["yellow"], "range": (0, 1)},
        },
    },
    "Memory & Language": {
        "description": "记忆与语言",
        "params": {
            "hc_memory_count":       {"label": "Memory Count",    "color": COLORS["blue"],   "range": (0, 500)},
            "hc_last_encoding_norm": {"label": "Encoding Norm",   "color": COLORS["sapphire"],"range": (0, 5)},
            "hc_retrieved_avg_reward":{"label": "Avg Reward",     "color": COLORS["green"],  "range": (-1, 1)},
            "language_valence":      {"label": "Lang. Valence",   "color": COLORS["teal"],   "range": (-1, 1)},
            "language_arousal":      {"label": "Lang. Arousal",   "color": COLORS["peach"],  "range": (0, 1)},
            "language_surprise":     {"label": "Lang. Surprise",  "color": COLORS["yellow"], "range": (0, 1)},
        },
    },
    "Behavioral Flags": {
        "description": "行为标志 (文本显示)",
        "text_keys": [
            "bsm_defense_behavior", "bsm_arousal_name", "allostatic_regime",
            "stress_type", "current_emotion", "limbic_emotion",
            "nt_state", "ans_polyvagal_state", "sleep_stage",
            "inferred_intent",
        ],
        "bool_keys": [
            "defensive_mode", "minimal_mode", "processing_throttle",
            "memory_pressure_critical", "memory_pressure_warning",
            "emergency_compression", "gc_triggered_by_waste",
            "sleep_request", "conservative_learning",
            "social_openness", "social_withdrawal",
            "encoding_suppressed", "high_self_awareness",
            "pfc_overrode_bg", "emotion_criticality",
            "is_introspective_mode", "should_ask_followup",
            "sleep_is_sleeping",
        ],
    },
    "LLM Parameters": {
        "description": "LLM 调用参数",
        "params": {
            "llm_temperature": {"label": "Temperature", "color": COLORS["peach"],  "range": (0, 2)},
            "llm_top_p":       {"label": "Top-P",       "color": COLORS["blue"],   "range": (0, 1)},
            "llm_max_tokens":  {"label": "Max Tokens",  "color": COLORS["green"],  "range": (0, 4096)},
        },
    },
}


# ═══════════════════════════════════════════════════════════════
# 单参数图表组件
# ═══════════════════════════════════════════════════════════════

class ParameterChart(QWidget):
    """单个参数的迷你折线图"""

    def __init__(self, key, config, max_points=120, parent=None):
        super().__init__(parent)
        self.key = key
        self.config = config
        self.history = collections.deque(maxlen=max_points)

        self.setFixedHeight(85)
        self.setMinimumWidth(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(1)

        # 标题行: 参数名 + 当前值
        header = QHBoxLayout()
        header.setSpacing(4)
        self.title_label = QLabel(config["label"])
        self.title_label.setStyleSheet(
            f"font-size: 10px; font-weight: bold; color: {config.get('color', CHART_FG)};"
        )
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #89b4fa;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(self.value_label)
        layout.addLayout(header)

        # Matplotlib 小图
        self.fig = Figure(figsize=(3, 0.55), dpi=80)
        self.fig.set_facecolor(CHART_BG)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet("background: transparent;")
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.fig.subplots_adjust(left=0.18, right=0.97, top=0.95, bottom=0.15)
        layout.addWidget(self.canvas, stretch=1)

    def _style_axes(self):
        self.ax.set_facecolor(CHART_BG)
        self.ax.tick_params(labelsize=6, colors=COLORS["surface2"], length=2)
        self.ax.grid(True, alpha=0.3, color=CHART_GRID, linewidth=0.5)
        for spine in self.ax.spines.values():
            spine.set_color(CHART_GRID)
            spine.set_linewidth(0.5)

    def update_value(self, value):
        if value is None:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return

        self.history.append(v)
        self.value_label.setText(f"{v:.3f}")

        # 重绘
        self.ax.clear()
        self._style_axes()

        data = list(self.history)
        if len(data) > 1:
            color = self.config.get("color", CHART_FG)
            self.ax.plot(data, color=color, linewidth=1.2, alpha=0.9)
            # 填充
            self.ax.fill_between(range(len(data)), data, alpha=0.1, color=color)

        rng = self.config.get("range")
        if rng:
            self.ax.set_ylim(rng)
        else:
            if data:
                mn, mx = min(data), max(data)
                margin = max((mx - mn) * 0.1, 0.01)
                self.ax.set_ylim(mn - margin, mx + margin)

        self.canvas.draw_idle()


# ═══════════════════════════════════════════════════════════════
# 数值参数选项卡
# ═══════════════════════════════════════════════════════════════

class NumericCategoryTab(QWidget):
    """包含多个 ParameterChart 的分类选项卡"""

    def __init__(self, params_config, columns=4, parent=None):
        super().__init__(parent)
        self.charts = {}

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {COLORS['crust']}; }}")

        container = QWidget()
        container.setStyleSheet(f"background: {COLORS['crust']};")
        grid = QGridLayout(container)
        grid.setSpacing(4)
        grid.setContentsMargins(6, 6, 6, 6)

        for i, (key, cfg) in enumerate(params_config.items()):
            chart = ParameterChart(key, cfg)
            self.charts[key] = chart
            row, col = divmod(i, columns)
            grid.addWidget(chart, row, col)

        # 填充剩余格子
        total = len(params_config)
        remainder = total % columns
        if remainder:
            for _ in range(columns - remainder):
                grid.addWidget(QWidget(), total // columns, (total % columns) + _)

        scroll.setWidget(container)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def update_values(self, state):
        for key, chart in self.charts.items():
            val = state.get(key)
            if val is not None and isinstance(val, (int, float)):
                chart.update_value(val)


# ═══════════════════════════════════════════════════════════════
# 文本/布尔参数选项卡
# ═══════════════════════════════════════════════════════════════

class TextCategoryTab(QWidget):
    """显示字符串和布尔参数"""

    def __init__(self, text_keys=None, bool_keys=None, parent=None):
        super().__init__(parent)
        self.text_labels = {}
        self.bool_labels = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        if text_keys:
            grp_text = QGroupBox("String Parameters")
            grp_text.setStyleSheet(f"""
                QGroupBox {{
                    color: {COLORS['text']}; font-size: 12px; font-weight: bold;
                    border: 1px solid {COLORS['surface0']}; border-radius: 4px;
                    margin-top: 8px; padding-top: 16px;
                }}
            """)
            grid_t = QGridLayout(grp_text)
            for i, key in enumerate(text_keys):
                row, col = divmod(i, 3)
                lbl = QLabel(f"{key}: --")
                lbl.setStyleSheet(f"font-size: 11px; color: {COLORS['text']}; padding: 3px;")
                lbl.setMinimumWidth(280)
                self.text_labels[key] = lbl
                grid_t.addWidget(lbl, row, col)
            layout.addWidget(grp_text)

        if bool_keys:
            grp_bool = QGroupBox("Boolean Flags")
            grp_bool.setStyleSheet(f"""
                QGroupBox {{
                    color: {COLORS['text']}; font-size: 12px; font-weight: bold;
                    border: 1px solid {COLORS['surface0']}; border-radius: 4px;
                    margin-top: 8px; padding-top: 16px;
                }}
            """)
            grid_b = QGridLayout(grp_bool)
            for i, key in enumerate(bool_keys):
                row, col = divmod(i, 3)
                lbl = QLabel(f"{key}: --")
                lbl.setStyleSheet(f"font-size: 11px; color: {COLORS['subtext0']}; padding: 3px;")
                lbl.setMinimumWidth(280)
                self.bool_labels[key] = lbl
                grid_b.addWidget(lbl, row, col)
            layout.addWidget(grp_bool)

        layout.addStretch()

    def update_values(self, state):
        for key, lbl in self.text_labels.items():
            val = state.get(key, "--")
            if isinstance(val, str):
                display = val
            elif isinstance(val, (list, dict)):
                display = str(val)[:60]
            else:
                display = str(val) if val is not None else "--"
            lbl.setText(f"{key}: {display}")

        for key, lbl in self.bool_labels.items():
            val = state.get(key)
            if val is True:
                lbl.setText(f"{key}: TRUE")
                lbl.setStyleSheet(
                    f"font-size: 11px; font-weight: bold; color: {COLORS['green']}; padding: 3px;"
                )
            elif val is False:
                lbl.setText(f"{key}: false")
                lbl.setStyleSheet(
                    f"font-size: 11px; color: {COLORS['surface2']}; padding: 3px;"
                )
            else:
                lbl.setText(f"{key}: --")
                lbl.setStyleSheet(
                    f"font-size: 11px; color: {COLORS['subtext0']}; padding: 3px;"
                )


# ═══════════════════════════════════════════════════════════════
# 演示数据生成器
# ═══════════════════════════════════════════════════════════════

class DemoStateGenerator:
    """在无 agent 时生成模拟数据"""

    def __init__(self):
        self.step = 0
        self.state = {}
        self._init_values()

    def _init_values(self):
        """为所有参数初始化基线值"""
        defaults = {
            "nt_dopamine": 0.5, "nt_serotonin": 0.5, "nt_acetylcholine": 0.4,
            "nt_norepinephrine": 0.4, "nt_motivation": 0.5, "nt_arousal": 0.5,
            "hormone_cortisol": 0.3, "hormone_adrenaline": 0.3,
            "hormone_melatonin": 0.3, "hormone_oxytocin": 0.3,
            "encoding_modulation": 1.0, "exploration_modulation": 1.0,
            "bsm_arousal": 0.6, "bsm_consciousness_gate": 0.8,
            "bsm_cortical_activation": 0.7, "bsm_respiratory_rate": 16,
            "bsm_heart_rate": 72, "bsm_blood_pressure": 120,
            "bsm_pain_gating": 0.5,
            "ans_sympathetic": 0.4, "ans_parasympathetic": 0.6,
            "ans_hrv": 0.6, "heart_rate": 72, "blood_pressure": 120,
            "cortisol_level": 0.3, "cortisol": 0.3,
            "hpa_crh": 0.3, "hpa_acth": 0.3,
            "brain_waste": 0.2, "neuroinflammation": 0.1,
            "myelination_level": 0.5, "brain_health": 0.8,
            "glymphatic_clearance": 0.6, "allostatic_load": 0.1,
            "regulatory_capacity": 0.8,
            "limbic_valence": 0.0, "limbic_arousal": 0.5,
            "limbic_emotional_attention": 0.5,
            "mood_valence": 0.0, "mood_arousal": 0.5,
            "pfc_inhibition": 0.3, "pfc_maturity": 0.5,
            "pfc_plan_depth": 2, "bg_td_error": 0.0,
            "bg_habit_strength": 0.2, "conscious_load": 0.3,
            "scn_alertness": 0.7, "scn_melatonin": 0.2,
            "scn_cortisol_rhythm": 0.5, "scn_wake_drive": 0.7,
            "scn_sleep_pressure": 0.3, "scn_temperature": 36.8,
            "sleep_fatigue": 0.2,
            "free_energy": 0.5, "prediction_error": 0.3,
            "precision": 0.7, "active_inference_drive": 0.3,
            "social_engagement": 0.5, "social_capacity": 0.5,
            "mirror_resonance": 0.3, "affective_empathy": 0.4,
            "cognitive_empathy": 0.5, "compassion": 0.3,
            "pain_resonance": 0.2, "empathy_level": 0.5,
            "self_coherence": 0.7, "self_endorsement": 0.7,
            "narrative_continuity": 0.7,
            "plasticity_bdnf": 0.5, "plasticity_synapses": 0.5,
            "info_gain": 0.1,
            "hc_memory_count": 10, "hc_last_encoding_norm": 1.0,
            "hc_retrieved_avg_reward": 0.3,
            "language_valence": 0.0, "language_arousal": 0.4,
            "language_surprise": 0.2,
            "llm_temperature": 0.7, "llm_top_p": 0.9, "llm_max_tokens": 2048,
        }
        self.state.update(defaults)

    @property
    def _internal_state(self):
        return self.state

    def tick(self):
        """每步微调参数，产生动态曲线"""
        self.step += 1
        t = self.step * 0.05

        # 神经递质: 缓慢波动
        for k in ["nt_dopamine", "nt_serotonin", "nt_acetylcholine", "nt_norepinephrine"]:
            base = self.state[k]
            noise = random.gauss(0, 0.02)
            wave = 0.03 * math.sin(t + hash(k) % 10)
            self.state[k] = max(0, min(1, base + noise + wave))

        self.state["nt_motivation"] = max(0, min(1,
            0.5 + 0.2 * math.sin(t * 0.7) + random.gauss(0, 0.01)))
        self.state["nt_arousal"] = max(0, min(1,
            0.5 + 0.15 * math.sin(t * 0.5 + 1) + random.gauss(0, 0.01)))

        # 激素
        self.state["hormone_cortisol"] = max(0, min(1,
            0.3 + 0.1 * math.sin(t * 0.3) + random.gauss(0, 0.01)))
        self.state["hormone_adrenaline"] = max(0, min(1,
            0.3 + 0.15 * math.sin(t * 0.4 + 2) + random.gauss(0, 0.01)))
        self.state["hormone_melatonin"] = max(0, min(1,
            0.3 + 0.2 * math.sin(t * 0.1) + random.gauss(0, 0.005)))
        self.state["hormone_oxytocin"] = max(0, min(1,
            0.3 + 0.1 * math.sin(t * 0.2 + 3) + random.gauss(0, 0.01)))

        # 脑干
        self.state["bsm_heart_rate"] = max(50, min(160,
            72 + 8 * math.sin(t * 0.8) + random.gauss(0, 1)))
        self.state["bsm_respiratory_rate"] = max(8, min(30,
            16 + 3 * math.sin(t * 0.6) + random.gauss(0, 0.5)))
        self.state["bsm_blood_pressure"] = max(80, min(180,
            120 + 10 * math.sin(t * 0.4) + random.gauss(0, 2)))
        self.state["heart_rate"] = self.state["bsm_heart_rate"]
        self.state["blood_pressure"] = self.state["bsm_blood_pressure"]
        self.state["bsm_arousal"] = max(0, min(1,
            0.6 + 0.15 * math.sin(t * 0.3) + random.gauss(0, 0.01)))
        self.state["bsm_consciousness_gate"] = max(0, min(1,
            0.8 + 0.05 * math.sin(t * 0.2) + random.gauss(0, 0.005)))
        self.state["bsm_cortical_activation"] = max(0, min(1,
            0.7 + 0.1 * math.sin(t * 0.5) + random.gauss(0, 0.01)))

        # 自主神经
        self.state["ans_sympathetic"] = max(0, min(1,
            0.4 + 0.15 * math.sin(t * 0.4) + random.gauss(0, 0.01)))
        self.state["ans_parasympathetic"] = max(0, min(1,
            1.0 - self.state["ans_sympathetic"] + random.gauss(0, 0.01)))
        self.state["ans_hrv"] = max(0, min(1,
            0.6 + 0.1 * math.sin(t * 0.3) + random.gauss(0, 0.01)))

        # HPA
        self.state["cortisol_level"] = self.state["hormone_cortisol"]
        self.state["cortisol"] = self.state["hormone_cortisol"]
        self.state["hpa_crh"] = max(0, min(1,
            0.3 + 0.1 * math.sin(t * 0.35) + random.gauss(0, 0.01)))
        self.state["hpa_acth"] = max(0, min(1,
            0.3 + 0.08 * math.sin(t * 0.25) + random.gauss(0, 0.01)))

        # 胶质
        self.state["brain_waste"] = max(0, min(1,
            0.2 + 0.05 * math.sin(t * 0.1) + random.gauss(0, 0.005)))
        self.state["neuroinflammation"] = max(0, min(1,
            0.1 + 0.03 * math.sin(t * 0.08) + random.gauss(0, 0.003)))
        self.state["brain_health"] = max(0, min(1,
            0.8 - 0.1 * self.state["neuroinflammation"] + random.gauss(0, 0.005)))

        # 边缘系统
        self.state["limbic_valence"] = max(-1, min(1,
            0.3 * math.sin(t * 0.2) + random.gauss(0, 0.02)))
        self.state["limbic_arousal"] = max(0, min(1,
            0.5 + 0.2 * math.sin(t * 0.35) + random.gauss(0, 0.01)))
        self.state["mood_valence"] = max(-1, min(1,
            0.2 * math.sin(t * 0.15) + random.gauss(0, 0.01)))
        self.state["mood_arousal"] = max(0, min(1,
            0.5 + 0.1 * math.sin(t * 0.25) + random.gauss(0, 0.008)))

        # 前额叶/基底节
        self.state["pfc_inhibition"] = max(0, min(1,
            0.3 + 0.1 * math.sin(t * 0.4) + random.gauss(0, 0.01)))
        self.state["bg_td_error"] = max(-1, min(1,
            0.2 * math.sin(t * 0.6) + random.gauss(0, 0.03)))
        self.state["conscious_load"] = max(0, min(1,
            0.3 + 0.15 * math.sin(t * 0.3) + random.gauss(0, 0.01)))

        # 昼夜
        hour = (t * 0.5) % 24
        self.state["scn_circadian_hour"] = hour
        self.state["scn_melatonin"] = max(0, min(1, 0.5 - 0.4 * math.cos(2 * math.pi * hour / 24)))
        self.state["scn_alertness"] = max(0, min(1, 0.5 + 0.4 * math.cos(2 * math.pi * hour / 24)))
        self.state["scn_temperature"] = 36.5 + 0.5 * math.sin(2 * math.pi * (hour - 6) / 24)

        # 预测编码
        self.state["free_energy"] = max(0, min(2,
            0.5 + 0.2 * math.sin(t * 0.3) + random.gauss(0, 0.02)))
        self.state["prediction_error"] = max(0, min(2,
            0.3 + 0.15 * math.sin(t * 0.45) + random.gauss(0, 0.02)))
        self.state["active_inference_drive"] = max(0, min(1,
            0.3 + 0.2 * math.sin(t * 0.2) + random.gauss(0, 0.01)))

        # 社会
        for k in ["social_engagement", "social_capacity", "mirror_resonance",
                   "affective_empathy", "cognitive_empathy", "compassion",
                   "pain_resonance", "empathy_level"]:
            base = self.state[k]
            self.state[k] = max(0, min(1, base + random.gauss(0, 0.005)))

        # 自我/可塑性
        self.state["plasticity_bdnf"] = max(0, min(1,
            0.5 + 0.1 * math.sin(t * 0.15) + random.gauss(0, 0.005)))
        self.state["self_coherence"] = max(0, min(1,
            0.7 + 0.05 * math.sin(t * 0.1) + random.gauss(0, 0.003)))

        # 记忆
        self.state["hc_memory_count"] = min(500, 10 + self.step // 5)
        self.state["language_valence"] = self.state["limbic_valence"]
        self.state["language_arousal"] = self.state["limbic_arousal"]

        # LLM 参数
        self.state["llm_temperature"] = max(0.1, min(2.0,
            0.7 + 0.2 * math.sin(t * 0.3) + random.gauss(0, 0.02)))
        self.state["llm_top_p"] = max(0.1, min(1.0,
            0.9 + 0.05 * math.sin(t * 0.2) + random.gauss(0, 0.005)))

        # 文本参数
        emotions = ["neutral", "joy", "calm", "curious", "surprise"]
        self.state["current_emotion"] = emotions[int(t * 0.2) % len(emotions)]
        self.state["limbic_emotion"] = self.state["current_emotion"]
        self.state["bsm_defense_behavior"] = "none"
        self.state["bsm_arousal_name"] = "alert" if self.state["bsm_arousal"] > 0.5 else "rest"
        self.state["allostatic_regime"] = "stable"
        self.state["stress_type"] = "none"
        self.state["nt_state"] = "balanced"
        self.state["ans_polyvagal_state"] = "ventral_vagal"
        self.state["sleep_stage"] = "awake"
        self.state["inferred_intent"] = "neutral"
        self.state["step"] = self.step

        # 布尔标志
        self.state["defensive_mode"] = self.state["allostatic_load"] > 0.8
        self.state["high_self_awareness"] = random.random() < 0.05
        self.state["social_openness"] = self.state["hormone_oxytocin"] > 0.6
        self.state["sleep_is_sleeping"] = False
        self.state["should_ask_followup"] = False


# ═══════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════

class NeuroMonitorWindow(QMainWindow):
    """Simulacrum 大脑参数实时监测面板"""

    def __init__(self, agent=None, interval=250, parent=None):
        super().__init__(parent)
        self.agent = agent
        self.demo = None if agent else DemoStateGenerator()
        self._paused = False
        self._interval = interval

        self.setWindowTitle("Simulacrum Neuro Monitor - Simulacrum")
        self.setMinimumSize(1280, 720)
        self.resize(1400, 800)

        self._init_ui()
        self._apply_theme()

        # 刷新定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh)
        self.timer.start(self._interval)

    # ── UI 构建 ──────────────────────────────────────────

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── 工具栏 ──
        self.toolbar = QToolBar("Main")
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)

        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setFixedWidth(80)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.toolbar.addWidget(self.pause_btn)

        self.toolbar.addSeparator()

        lbl = QLabel(" Refresh: ")
        lbl.setStyleSheet(f"color: {SUBTEXT_COLOR}; font-size: 11px;")
        self.toolbar.addWidget(lbl)
        self.refresh_combo = QComboBox()
        self.refresh_combo.addItems(["100ms", "250ms", "500ms", "1s", "2s"])
        self.refresh_combo.setCurrentIndex(1)
        self.refresh_combo.setFixedWidth(80)
        self.refresh_combo.currentTextChanged.connect(self._change_refresh_rate)
        self.toolbar.addWidget(self.refresh_combo)

        self.toolbar.addSeparator()

        self.step_label = QLabel("  Step: 0")
        self.step_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 11px;")
        self.toolbar.addWidget(self.step_label)

        self.emotion_label = QLabel("  Emotion: neutral")
        self.emotion_label.setStyleSheet(f"color: {COLORS['yellow']}; font-size: 11px;")
        self.toolbar.addWidget(self.emotion_label)

        self.valence_label = QLabel("  Valence: 0.000")
        self.valence_label.setStyleSheet(f"color: {COLORS['green']}; font-size: 11px;")
        self.toolbar.addWidget(self.valence_label)

        self.arousal_label = QLabel("  Arousal: 0.500")
        self.arousal_label.setStyleSheet(f"color: {COLORS['peach']}; font-size: 11px;")
        self.toolbar.addWidget(self.arousal_label)

        spacer = QLabel()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(spacer)

        mode_lbl = QLabel("DEMO " if not self.agent else "LIVE ")
        mode_lbl.setStyleSheet(
            f"color: {COLORS['red'] if not self.agent else COLORS['green']}; "
            f"font-size: 11px; font-weight: bold; padding-right: 8px;"
        )
        self.toolbar.addWidget(mode_lbl)

        self.addToolBar(self.toolbar)

        # ── 选项卡 ──
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.category_tabs = {}
        self.text_tab = None

        for cat_name, cat_def in CATEGORY_DEFS.items():
            if "params" in cat_def:
                tab = NumericCategoryTab(cat_def["params"], columns=4)
                self.tabs.addTab(tab, f"{cat_name} ({len(cat_def['params'])})")
                self.category_tabs[cat_name] = tab
            elif "text_keys" in cat_def or "bool_keys" in cat_def:
                self.text_tab = TextCategoryTab(
                    text_keys=cat_def.get("text_keys", []),
                    bool_keys=cat_def.get("bool_keys", []),
                )
                self.tabs.addTab(self.text_tab, cat_name)

        main_layout.addWidget(self.tabs)

        # ── 状态栏 ──
        self.statusBar().showMessage("Ready")
        self.statusBar().setFixedHeight(22)

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {COLORS['base']};
            }}
            QToolBar {{
                background: {COLORS['mantle']};
                border-bottom: 1px solid {COLORS['surface0']};
                spacing: 4px;
                padding: 2px;
            }}
            QPushButton {{
                background: {COLORS['surface0']};
                color: {TEXT_COLOR};
                border: 1px solid {COLORS['surface1']};
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: {COLORS['surface1']};
            }}
            QPushButton:checked {{
                background: {COLORS['red']};
                color: {COLORS['base']};
            }}
            QComboBox {{
                background: {COLORS['surface0']};
                color: {TEXT_COLOR};
                border: 1px solid {COLORS['surface1']};
                padding: 2px 6px;
                font-size: 11px;
                border-radius: 3px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background: {COLORS['mantle']};
                color: {TEXT_COLOR};
                selection-background-color: {COLORS['surface1']};
            }}
            QTabWidget::pane {{
                border: none;
                background: {COLORS['crust']};
            }}
            QTabBar::tab {{
                background: {COLORS['mantle']};
                color: {SUBTEXT_COLOR};
                padding: 6px 14px;
                border: 1px solid {COLORS['surface0']};
                border-bottom: none;
                margin-right: 1px;
                font-size: 11px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['crust']};
                color: {TEXT_COLOR};
                font-weight: bold;
            }}
            QTabBar::tab:hover {{
                background: {COLORS['surface0']};
            }}
            QStatusBar {{
                background: {COLORS['mantle']};
                color: {SUBTEXT_COLOR};
                font-size: 10px;
                border-top: 1px solid {COLORS['surface0']};
            }}
            QScrollArea {{
                border: none;
            }}
            QScrollBar:vertical {{
                background: {COLORS['mantle']};
                width: 8px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['surface1']};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QGroupBox {{
                color: {TEXT_COLOR};
                font-size: 12px;
                font-weight: bold;
                border: 1px solid {COLORS['surface0']};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 16px;
            }}
        """)

    # ── 刷新逻辑 ──────────────────────────────────────────

    def _refresh(self):
        if self._paused:
            return

        if self.agent:
            state = self.agent._internal_state
        elif self.demo:
            self.demo.tick()
            state = self.demo._internal_state
        else:
            return

        # 更新所有数值选项卡
        for tab in self.category_tabs.values():
            tab.update_values(state)

        # 更新文本选项卡
        if self.text_tab:
            self.text_tab.update_values(state)

        # 更新状态栏
        step = state.get("step", 0)
        emotion = state.get("current_emotion", state.get("limbic_emotion", "--"))
        valence = state.get("limbic_valence", 0)
        arousal = state.get("limbic_arousal", 0)

        self.step_label.setText(f"  Step: {step}")
        self.emotion_label.setText(f"  Emotion: {emotion}")
        self.valence_label.setText(f"  Valence: {valence:.3f}")
        self.arousal_label.setText(f"  Arousal: {arousal:.3f}")

    def _toggle_pause(self):
        self._paused = self.pause_btn.isChecked()
        self.pause_btn.setText("▶ Resume" if self._paused else "⏸ Pause")
        self.statusBar().showMessage("Paused" if self._paused else "Running")

    def _change_refresh_rate(self, text):
        rate_map = {"100ms": 100, "250ms": 250, "500ms": 500, "1s": 1000, "2s": 2000}
        self._interval = rate_map.get(text, 250)
        self.timer.setInterval(self._interval)
        self.statusBar().showMessage(f"Refresh rate: {text}")


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Simulacrum Neuro Monitor")
    parser.add_argument("--agent", action="store_true",
                        help="Connect to a live Simulacrum agent (use with chat_main.py)")
    parser.add_argument("--interval", type=int, default=250,
                        help="Refresh interval in ms (default: 250)")
    parser.add_argument("--provider", default="mock",
                        help="LLM provider for agent (only with --agent)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    agent = None
    if args.agent:
        from simulacrum.utils.config import load_config
        from simulacrum.core.agent import Simulacrum

        config = load_config(
            llm_provider=args.provider,
            device=args.device,
            initial_balance=100.0,
        )
        print("[INIT] Creating Simulacrum agent...")
        agent = Simulacrum(config=config)
        print("[OK] Agent ready")

    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = NeuroMonitorWindow(agent=agent, interval=args.interval)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
