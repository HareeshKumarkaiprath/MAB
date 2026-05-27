# =============================================================================
# BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, PILANI — WILP
# Deep Reinforcement Learning — Lab Assignment 1
# Part 1: Multi-Armed Bandit (MAB)
# Group 218 — Interactive Treatment Recommendation UI
#
# Run with:  python Team_218_MAB_UI.py
# Requires:  pip install PyQt5 matplotlib numpy pandas
# =============================================================================

import sys
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QCheckBox, QComboBox,
    QSpinBox, QGroupBox, QScrollArea, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QFrame, QSizePolicy,
    QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

# =============================================================================
# GROUP 218 CONSTANTS
# =============================================================================
G            = 218
K            = (G % 3) + 5                                        # 7 medicines
TRUE_PROBS   = [round(0.4 + ((G + i) % 6) * 0.07, 2) for i in range(K)]
MED_COLORS   = ["#4682b4","#e87320","#2e8b57","#d4a800",
                "#c0392b","#8a2be2","#008b8b"]
STRATEGY_MAP = {
    "Greedy (warm-up + exploit)" : "greedy",
    "ε-Greedy  1%  exploration"  : "eps1",
    "ε-Greedy 10%  exploration"  : "eps10",
    "ε-Greedy 50%  exploration"  : "eps50",
    "UCB1 (confidence-based)"    : "ucb1",
    "Run ALL strategies"         : "all",
}
STRAT_COLORS = {
    "greedy":"#4682b4","eps1":"#c0392b",
    "eps10" :"#2e8b57","eps50":"#e87320","ucb1":"#8a2be2"
}
STRAT_LABELS = {
    "greedy":"Greedy","eps1":"ε=1%",
    "eps10" :"ε=10%","eps50":"ε=50%","ucb1":"UCB1"
}

# =============================================================================
# SIMULATION ENGINE  (pure Python — no Qt dependency)
# =============================================================================

def seeded_rng(seed: int):
    """Returns a simple LCG pseudo-random generator for reproducibility."""
    state = [seed & 0xFFFFFFFF]
    def rng():
        state[0] = (1664525 * state[0] + 1013904223) & 0xFFFFFFFF
        return state[0] / 4294967296.0
    return rng


def run_simulation(strategy: str, active_meds: list, n_patients: int,
                   warmup: int, seed: int) -> dict:
    """
    Runs one MAB strategy simulation.

    Parameters
    ----------
    strategy    : str   - one of greedy / eps1 / eps10 / eps50 / ucb1
    active_meds : list  - indices of medicines to include
    n_patients  : int   - total patients to simulate
    warmup      : int   - warm-up trials per medicine (Greedy only)
    seed        : int   - random seed for reproducibility

    Returns
    -------
    dict with keys:
        counts       - {med_idx: int}    selection counts
        avgs         - {med_idx: float}  average utility per medicine
        cum_rewards  - list[float]       cumulative reward at each patient step
        final_reward - float             total cumulative reward
        dataset      - pd.DataFrame      full patient-level dataset
    """
    rng     = seeded_rng(seed)
    counts  = {m: 0 for m in active_meds}
    totals  = {m: 0.0 for m in active_meds}
    avgs    = {m: 0.0 for m in active_meds}
    records = []
    cum     = 0.0
    cum_rewards = []

    for pid in range(n_patients):
        severity = (pid % 5) + 1

        # ── Medicine selection ────────────────────────────────────────────────
        if strategy == "greedy":
            if pid < warmup * len(active_meds):
                med = active_meds[pid % len(active_meds)]
            else:
                med = max(active_meds, key=lambda m: avgs[m])

        elif strategy == "eps1":
            if all(counts[m] == 0 for m in active_meds) or rng() < 0.01:
                med = active_meds[int(rng() * len(active_meds))]
            else:
                med = max(active_meds, key=lambda m: avgs[m])

        elif strategy == "eps10":
            if all(counts[m] == 0 for m in active_meds) or rng() < 0.10:
                med = active_meds[int(rng() * len(active_meds))]
            else:
                med = max(active_meds, key=lambda m: avgs[m])

        elif strategy == "eps50":
            if all(counts[m] == 0 for m in active_meds) or rng() < 0.50:
                med = active_meds[int(rng() * len(active_meds))]
            else:
                med = max(active_meds, key=lambda m: avgs[m])

        elif strategy == "ucb1":
            zeros = [m for m in active_meds if counts[m] == 0]
            if zeros:
                med = zeros[0]
            else:
                t = pid + 1
                med = max(active_meds,
                          key=lambda m: avgs[m] + math.sqrt(2 * math.log(t) / counts[m]))

        # ── Simulate treatment ────────────────────────────────────────────────
        outcome = 1 if rng() < TRUE_PROBS[med] else 0
        utility = outcome * (1 - severity / 10.0)

        counts[med] += 1
        totals[med] += utility
        avgs[med]    = totals[med] / counts[med]
        cum         += utility
        cum_rewards.append(round(cum, 4))

        records.append({
            "patient_id"        : pid,
            "severity_score"    : severity,
            "assigned_medicine" : med,
            "clinical_outcome"  : outcome,
            "utility_score"     : round(utility, 4),
            "cumulative_reward" : round(cum, 4),
        })

    df = pd.DataFrame(records)
    return {
        "counts"      : counts,
        "avgs"        : avgs,
        "cum_rewards" : cum_rewards,
        "final_reward": round(cum, 2),
        "dataset"     : df,
    }


# =============================================================================
# WORKER THREAD — keeps UI responsive during simulation
# =============================================================================

class SimWorker(QThread):
    """Runs simulations in a background thread and emits results when done."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)

    def __init__(self, strategies, active_meds, n_patients, warmup, seed):
        super().__init__()
        self.strategies  = strategies
        self.active_meds = active_meds
        self.n_patients  = n_patients
        self.warmup      = warmup
        self.seed        = seed

    def run(self):
        results = {}
        for i, strat in enumerate(self.strategies):
            results[strat] = run_simulation(
                strat, self.active_meds, self.n_patients,
                self.warmup, self.seed + i
            )
            self.progress.emit(int((i + 1) / len(self.strategies) * 100))
        self.finished.emit(results)


# =============================================================================
# MATPLOTLIB CANVAS WIDGET
# =============================================================================

class MplCanvas(FigureCanvas):
    """Embeddable Matplotlib figure for PyQt5."""
    def __init__(self, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# =============================================================================
# MEDICINE SELECTOR CARD WIDGET
# =============================================================================

class MedCard(QFrame):
    """Clickable card representing one medicine with toggle behaviour."""

    toggled = pyqtSignal(int, bool)   # (med_index, is_selected)

    def __init__(self, med_idx: int, parent=None):
        super().__init__(parent)
        self.med_idx    = med_idx
        self.is_selected = True
        self._setup_ui()
        self._apply_style(True)

    def _setup_ui(self):
        self.setFixedSize(110, 130)
        self.setCursor(Qt.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        # Colour swatch circle (simulated with label background)
        swatch = QLabel()
        swatch.setFixedSize(22, 22)
        color = MED_COLORS[self.med_idx]
        swatch.setStyleSheet(
            f"background:{color};border-radius:11px;border:none")
        layout.addWidget(swatch)

        # Medicine name
        name_lbl = QLabel(f"Medicine {self.med_idx}")
        name_lbl.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(name_lbl)

        # Probability
        prob = TRUE_PROBS[self.med_idx]
        prob_lbl = QLabel(f"P = {prob:.2f}")
        prob_lbl.setFont(QFont("Arial", 9))
        prob_lbl.setStyleSheet("color:#666")
        layout.addWidget(prob_lbl)

        # Best badge
        best_prob = max(TRUE_PROBS)
        if prob == best_prob:
            badge = QLabel("★ best arm")
            badge.setFont(QFont("Arial", 8, QFont.Bold))
            badge.setStyleSheet(
                "color:#27500A;background:#EAF3DE;"
                "border-radius:8px;padding:2px 6px")
            layout.addWidget(badge)
        layout.addStretch()

        # Checkbox (purely visual indicator, click handled on frame)
        self.chk = QCheckBox()
        self.chk.setChecked(True)
        self.chk.setAttribute(Qt.WA_TransparentForMouseEvents)
        layout.addWidget(self.chk)

    def _apply_style(self, selected: bool):
        if selected:
            self.setStyleSheet(
                "MedCard{border:2px solid #185FA5;border-radius:10px;"
                "background:#E6F1FB}"
                "MedCard:hover{background:#d0e8f8}")
        else:
            self.setStyleSheet(
                "MedCard{border:1px solid #ccc;border-radius:10px;"
                "background:#f8f8f8}"
                "MedCard:hover{background:#efefef}")

    def mousePressEvent(self, event):
        self.is_selected = not self.is_selected
        self.chk.setChecked(self.is_selected)
        self._apply_style(self.is_selected)
        self.toggled.emit(self.med_idx, self.is_selected)


# =============================================================================
# MAIN WINDOW
# =============================================================================

class MABApp(QMainWindow):
    """Main application window for the MAB Treatment Recommendation System."""

    def __init__(self):
        super().__init__()
        self.selected_meds = set(range(K))
        self.results       = {}
        self.worker        = None
        self._setup_window()
        self._build_ui()

    # ── Window setup ──────────────────────────────────────────────────────────

    def _setup_window(self):
        self.setWindowTitle(
            "MAB Treatment Recommendation System — Group 218 | BITS Pilani WILP")
        self.setMinimumSize(1100, 780)
        self.resize(1280, 860)

    # ── Master layout ─────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QLabel(
            "Adaptive Treatment Recommendation System  ·  Group 218  ·  K = 7 Medicines")
        hdr.setFont(QFont("Arial", 15, QFont.Bold))
        hdr.setStyleSheet("color:#0C447C;padding:4px 0")
        root.addWidget(hdr)

        sub = QLabel(
            f"G = {G}  |  K = {K}  |  True probabilities: "
            + "  ".join(f"M{i}={p:.2f}" for i, p in enumerate(TRUE_PROBS)))
        sub.setFont(QFont("Arial", 9))
        sub.setStyleSheet("color:#555")
        root.addWidget(sub)

        # ── Splitter: left config panel / right results tabs ──────────────────
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_config_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setSizes([380, 860])

    # ── Left: configuration panel ─────────────────────────────────────────────

    def _build_config_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        # Medicine selector group
        med_grp = QGroupBox("Step 1 — Select medicines")
        med_grp.setFont(QFont("Arial", 10, QFont.Bold))
        mg_layout = QVBoxLayout(med_grp)

        info = QLabel(
            "Click cards to include / exclude medicines from the simulation.")
        info.setWordWrap(True)
        info.setStyleSheet(
            "background:#E6F1FB;color:#0C447C;border-radius:6px;"
            "padding:7px 10px;font-size:11px")
        mg_layout.addWidget(info)

        # Medicine cards grid
        cards_widget = QWidget()
        cards_layout = QGridLayout(cards_widget)
        cards_layout.setSpacing(8)
        self.med_cards = []
        for i in range(K):
            card = MedCard(i)
            card.toggled.connect(self._on_med_toggled)
            self.med_cards.append(card)
            cards_layout.addWidget(card, i // 4, i % 4)
        mg_layout.addWidget(cards_widget)

        # Select all / clear buttons
        btn_row = QHBoxLayout()
        sel_all = QPushButton("Select all")
        sel_all.clicked.connect(self._select_all_meds)
        sel_all.setStyleSheet(self._btn_style("#185FA5"))
        clr_btn = QPushButton("Clear all")
        clr_btn.clicked.connect(self._clear_all_meds)
        clr_btn.setStyleSheet(self._btn_style("#888"))
        self.sel_count_lbl = QLabel(f"{K} selected")
        self.sel_count_lbl.setStyleSheet("color:#555;font-size:11px")
        btn_row.addWidget(sel_all)
        btn_row.addWidget(clr_btn)
        btn_row.addWidget(self.sel_count_lbl)
        btn_row.addStretch()
        mg_layout.addLayout(btn_row)
        layout.addWidget(med_grp)

        # Strategy config group
        cfg_grp = QGroupBox("Step 2 — Configure simulation")
        cfg_grp.setFont(QFont("Arial", 10, QFont.Bold))
        cfg_layout = QGridLayout(cfg_grp)
        cfg_layout.setSpacing(8)

        cfg_layout.addWidget(QLabel("Strategy:"), 0, 0)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(list(STRATEGY_MAP.keys()))
        self.strategy_combo.setCurrentIndex(2)       # ε-Greedy 10% default
        cfg_layout.addWidget(self.strategy_combo, 0, 1)

        cfg_layout.addWidget(QLabel("Patients:"), 1, 0)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(100, 5000)
        self.n_spin.setSingleStep(100)
        self.n_spin.setValue(1000)
        cfg_layout.addWidget(self.n_spin, 1, 1)

        cfg_layout.addWidget(QLabel("Warm-up / medicine (Greedy):"), 2, 0)
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(1, 50)
        self.warmup_spin.setValue(10)
        cfg_layout.addWidget(self.warmup_spin, 2, 1)

        cfg_layout.addWidget(QLabel("Random seed:"), 3, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 99999)
        self.seed_spin.setValue(218)
        cfg_layout.addWidget(self.seed_spin, 3, 1)

        layout.addWidget(cfg_grp)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar{border:1px solid #ccc;border-radius:5px;height:16px}"
            "QProgressBar::chunk{background:#185FA5;border-radius:4px}")
        layout.addWidget(self.progress_bar)

        # Run button
        self.run_btn = QPushButton("  Run simulation & generate report")
        self.run_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.run_btn.setMinimumHeight(44)
        self.run_btn.setStyleSheet(self._btn_style("#185FA5", large=True))
        self.run_btn.clicked.connect(self._run_simulation)
        layout.addWidget(self.run_btn)

        # Export button
        self.export_btn = QPushButton("  Export CSV datasets")
        self.export_btn.setFont(QFont("Arial", 10))
        self.export_btn.setMinimumHeight(36)
        self.export_btn.setStyleSheet(self._btn_style("#2e8b57"))
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_csv)
        layout.addWidget(self.export_btn)

        layout.addStretch()
        return panel

    # ── Right: results panel with tabs ────────────────────────────────────────

    def _build_results_panel(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Arial", 10))
        self.tabs.setStyleSheet(
            "QTabBar::tab{padding:8px 18px;font-size:11px}"
            "QTabBar::tab:selected{color:#185FA5;border-bottom:2px solid #185FA5}")

        # Tab 1 — Summary
        self.summary_widget = QWidget()
        self.summary_layout = QVBoxLayout(self.summary_widget)
        self._build_summary_placeholder()
        self.tabs.addTab(self.summary_widget, "Summary")

        # Tab 2 — Cumulative reward chart
        self.cum_canvas = MplCanvas(width=8, height=4)
        self.tabs.addTab(self.cum_canvas, "Cumulative reward")

        # Tab 3 — Medicine frequency chart
        self.freq_canvas = MplCanvas(width=8, height=4)
        self.tabs.addTab(self.freq_canvas, "Medicine frequency")

        # Tab 4 — Avg utility chart
        self.util_canvas = MplCanvas(width=8, height=4)
        self.tabs.addTab(self.util_canvas, "Avg utility")

        # Tab 5 — Detailed data table
        self.table_widget = QTableWidget()
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        scroll = QScrollArea()
        scroll.setWidget(self.table_widget)
        scroll.setWidgetResizable(True)
        self.tabs.addTab(scroll, "Data table")

        return self.tabs

    def _build_summary_placeholder(self):
        """Placeholder shown before first simulation run."""
        for i in reversed(range(self.summary_layout.count())):
            self.summary_layout.itemAt(i).widget().deleteLater()
        ph = QLabel(
            "Configure and run a simulation to see results here.")
        ph.setAlignment(Qt.AlignCenter)
        ph.setStyleSheet("color:#aaa;font-size:14px;padding:60px")
        self.summary_layout.addWidget(ph)

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_med_toggled(self, med_idx: int, is_selected: bool):
        if is_selected:
            self.selected_meds.add(med_idx)
        else:
            self.selected_meds.discard(med_idx)
        self.sel_count_lbl.setText(f"{len(self.selected_meds)} selected")

    def _select_all_meds(self):
        self.selected_meds = set(range(K))
        for card in self.med_cards:
            card.is_selected = True
            card.chk.setChecked(True)
            card._apply_style(True)
        self.sel_count_lbl.setText(f"{K} selected")

    def _clear_all_meds(self):
        self.selected_meds = set()
        for card in self.med_cards:
            card.is_selected = False
            card.chk.setChecked(False)
            card._apply_style(False)
        self.sel_count_lbl.setText("0 selected")

    def _run_simulation(self):
        if len(self.selected_meds) < 2:
            QMessageBox.warning(self, "Selection error",
                                "Please select at least 2 medicines.")
            return

        strat_key = STRATEGY_MAP[self.strategy_combo.currentText()]
        strategies = list(STRAT_LABELS.keys()) if strat_key == "all" \
                     else [strat_key]
        active_meds = sorted(self.selected_meds)
        n_patients  = self.n_spin.value()
        warmup      = self.warmup_spin.value()
        seed        = self.seed_spin.value()

        self.run_btn.setEnabled(False)
        self.run_btn.setText("  Running simulation…")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.worker = SimWorker(strategies, active_meds, n_patients, warmup, seed)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_results)
        self.worker.start()

    def _on_results(self, results: dict):
        self.results = results
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("  Re-run simulation")
        self.export_btn.setEnabled(True)

        active_meds = sorted(self.selected_meds)
        n_patients  = self.n_spin.value()

        self._render_summary(results, active_meds)
        self._render_cumulative_chart(results, n_patients)
        self._render_frequency_chart(results, active_meds)
        self._render_utility_chart(results, active_meds)
        self._render_data_table(results, active_meds)

        self.tabs.setCurrentIndex(0)

    # ── Render: Summary tab ───────────────────────────────────────────────────

    def _render_summary(self, results: dict, active_meds: list):
        # Clear existing widgets
        for i in reversed(range(self.summary_layout.count())):
            w = self.summary_layout.itemAt(i).widget()
            if w:
                w.deleteLater()

        best_strat = max(results, key=lambda s: results[s]["final_reward"])
        best_med   = max(active_meds,
                         key=lambda m: results[best_strat]["avgs"].get(m, 0))

        # ── Metric cards row ──────────────────────────────────────────────────
        metric_row = QHBoxLayout()
        metrics = [
            ("Patients simulated",
             str(self.n_spin.value()),
             "Group 218"),
            ("Best strategy",
             STRAT_LABELS[best_strat],
             f"{results[best_strat]['final_reward']:.1f} reward"),
            ("Best medicine found",
             f"M{best_med}",
             f"P = {TRUE_PROBS[best_med]:.2f}"),
            ("Medicines included",
             str(len(active_meds)),
             f"of {K} available"),
        ]
        for label, value, sub in metrics:
            card = self._metric_card(label, value, sub)
            metric_row.addWidget(card)
        self.summary_layout.addLayout(metric_row)

        # ── Ranked strategy table ─────────────────────────────────────────────
        tbl_label = QLabel("Strategy ranking")
        tbl_label.setFont(QFont("Arial", 11, QFont.Bold))
        tbl_label.setStyleSheet("color:#0C447C;padding:8px 0 4px 0")
        self.summary_layout.addWidget(tbl_label)

        strats_sorted = sorted(results,
                               key=lambda s: results[s]["final_reward"],
                               reverse=True)
        tbl = QTableWidget(len(strats_sorted), 6)
        tbl.setHorizontalHeaderLabels([
            "Rank", "Strategy", "Final reward",
            "Best arm found", "Avg utility", "Recovery rate"])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setAlternatingRowColors(True)
        tbl.setMaximumHeight(240)

        for row, strat in enumerate(strats_sorted):
            r = results[strat]
            bm = max(active_meds, key=lambda m: r["avgs"].get(m, 0))
            avg_u = r["dataset"]["utility_score"].mean()
            rec   = r["dataset"]["clinical_outcome"].mean()

            items = [
                str(row + 1),
                STRAT_LABELS[strat],
                f"{r['final_reward']:.2f}",
                f"M{bm}  (P={TRUE_PROBS[bm]:.2f})",
                f"{avg_u:.4f}",
                f"{rec*100:.1f}%",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if strat == best_strat:
                    item.setBackground(QColor("#EAF3DE"))
                    item.setForeground(QColor("#27500A"))
                tbl.setItem(row, col, item)

        self.summary_layout.addWidget(tbl)

        # ── Per-medicine summary table ────────────────────────────────────────
        med_label = QLabel("Per-medicine results")
        med_label.setFont(QFont("Arial", 11, QFont.Bold))
        med_label.setStyleSheet("color:#0C447C;padding:8px 0 4px 0")
        self.summary_layout.addWidget(med_label)

        cols = ["Medicine", "True P"] + \
               [f"Count ({STRAT_LABELS[s]})" for s in strats_sorted] + \
               [f"Avg util ({STRAT_LABELS[s]})" for s in strats_sorted]
        tbl2 = QTableWidget(len(active_meds), len(cols))
        tbl2.setHorizontalHeaderLabels(cols)
        tbl2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl2.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl2.setAlternatingRowColors(True)

        best_prob = max(TRUE_PROBS[m] for m in active_meds)
        for row, med in enumerate(active_meds):
            is_best_med = TRUE_PROBS[med] == best_prob
            values = [f"M{med}", f"{TRUE_PROBS[med]:.2f}"] + \
                     [str(results[s]["counts"].get(med, 0)) for s in strats_sorted] + \
                     [f"{results[s]['avgs'].get(med, 0):.4f}" for s in strats_sorted]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if is_best_med:
                    item.setBackground(QColor("#FFF8E1"))
                    item.setForeground(QColor("#5a3800"))
                tbl2.setItem(row, col, item)

        scroll2 = QScrollArea()
        scroll2.setWidget(tbl2)
        scroll2.setWidgetResizable(True)
        scroll2.setMinimumHeight(200)
        self.summary_layout.addWidget(scroll2)
        self.summary_layout.addStretch()

    # ── Render: Cumulative reward chart ───────────────────────────────────────

    def _render_cumulative_chart(self, results: dict, n_patients: int):
        fig = self.cum_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)

        step = max(1, n_patients // 300)
        xs   = list(range(1, n_patients + 1, step))

        line_styles = {
            "greedy": "-","eps1":"--","eps10":"-","eps50":"--","ucb1":"-."
        }
        for strat, res in results.items():
            ys = [res["cum_rewards"][i - 1] for i in xs if i <= len(res["cum_rewards"])]
            xs2 = xs[:len(ys)]
            ax.plot(xs2, ys,
                    color=STRAT_COLORS[strat],
                    linestyle=line_styles.get(strat, "-"),
                    linewidth=2,
                    label=f"{STRAT_LABELS[strat]}  ({res['final_reward']:.1f})")

        ax.set_xlabel("Number of patients", fontsize=11)
        ax.set_ylabel("Cumulative utility score", fontsize=11)
        ax.set_title(
            f"Cumulative reward vs patients  |  Group {G}  |  K={K} medicines",
            fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines[["top","right"]].set_visible(False)
        self.cum_canvas.draw()

    # ── Render: Medicine frequency chart ─────────────────────────────────────

    def _render_frequency_chart(self, results: dict, active_meds: list):
        fig = self.freq_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)

        strats   = list(results.keys())
        n_strats = len(strats)
        n_meds   = len(active_meds)
        bar_w    = 0.7 / n_strats
        xs       = np.arange(n_meds)

        for i, strat in enumerate(strats):
            counts = [results[strat]["counts"].get(m, 0) for m in active_meds]
            offset = (i - n_strats / 2 + 0.5) * bar_w
            ax.bar(xs + offset, counts,
                   width=bar_w * 0.9,
                   color=STRAT_COLORS[strat],
                   label=STRAT_LABELS[strat],
                   alpha=0.85)

        ax.set_xticks(xs)
        ax.set_xticklabels([f"M{m}\nP={TRUE_PROBS[m]:.2f}" for m in active_meds],
                           fontsize=9)
        ax.set_xlabel("Medicine", fontsize=11)
        ax.set_ylabel("Number of patients assigned", fontsize=11)
        ax.set_title("Medicine selection frequency by strategy",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top","right"]].set_visible(False)
        self.freq_canvas.draw()

    # ── Render: Avg utility chart ─────────────────────────────────────────────

    def _render_utility_chart(self, results: dict, active_meds: list):
        fig = self.util_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)

        strats   = list(results.keys())
        n_strats = len(strats)
        xs       = np.arange(len(active_meds))
        bar_w    = 0.6 / n_strats

        for i, strat in enumerate(strats):
            avgs = [results[strat]["avgs"].get(m, 0) for m in active_meds]
            offset = (i - n_strats / 2 + 0.5) * bar_w
            ax.bar(xs + offset, avgs,
                   width=bar_w * 0.9,
                   color=STRAT_COLORS[strat],
                   label=STRAT_LABELS[strat],
                   alpha=0.82)

        # True probability overlay line
        true_p = [TRUE_PROBS[m] for m in active_meds]
        ax.plot(xs, true_p, "o--",
                color="#e87320", linewidth=2, markersize=6,
                label="True prob P_i")

        ax.set_xticks(xs)
        ax.set_xticklabels([f"M{m}" for m in active_meds], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Medicine", fontsize=11)
        ax.set_ylabel("Average utility (learned)", fontsize=11)
        ax.set_title(
            "Learned avg utility per medicine vs true probability",
            fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top","right"]].set_visible(False)
        self.util_canvas.draw()

    # ── Render: Data table tab ────────────────────────────────────────────────

    def _render_data_table(self, results: dict, active_meds: list):
        # Show dataset from the first (or only) strategy
        strat    = list(results.keys())[0]
        df       = results[strat]["dataset"]
        n_rows   = min(len(df), 500)      # cap display at 500 rows for speed
        df_disp  = df.head(n_rows)

        self.table_widget.clear()
        self.table_widget.setRowCount(n_rows)
        self.table_widget.setColumnCount(len(df_disp.columns))
        self.table_widget.setHorizontalHeaderLabels(list(df_disp.columns))

        for r_idx, row in enumerate(df_disp.itertuples(index=False)):
            for c_idx, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                # Highlight recovered patients
                if df_disp.columns[c_idx] == "clinical_outcome" and val == 1:
                    item.setBackground(QColor("#EAF3DE"))
                self.table_widget.setItem(r_idx, c_idx, item)

        note = QLabel(
            f"Showing first {n_rows} of {len(df)} rows  ·  "
            f"Strategy: {STRAT_LABELS[strat]}  ·  "
            f"Use 'Export CSV' to save all data")
        note.setStyleSheet(
            "color:#555;font-size:11px;padding:4px 0")

    # ── Export CSV ────────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self.results:
            return
        saved = []
        for strat, res in self.results.items():
            fname = f"Team_218_dataset_{strat}.csv"
            res["dataset"].to_csv(fname, index=False)
            saved.append(fname)

        # Summary CSV
        rows = []
        active_meds = sorted(self.selected_meds)
        for strat, res in self.results.items():
            bm   = max(active_meds, key=lambda m: res["avgs"].get(m, 0))
            avg_u = res["dataset"]["utility_score"].mean()
            rec   = res["dataset"]["clinical_outcome"].mean()
            rows.append({
                "strategy"               : STRAT_LABELS[strat],
                "final_cumulative_reward": res["final_reward"],
                "avg_utility_per_patient": round(avg_u, 4),
                "recovery_rate"          : round(rec, 4),
                "most_used_medicine"     : bm,
                "total_patients"         : len(res["dataset"]),
            })
        summary_df = pd.DataFrame(rows).sort_values(
            "final_cumulative_reward", ascending=False)
        summary_df.to_csv("Team_218_summary_all_strategies.csv", index=False)
        saved.append("Team_218_summary_all_strategies.csv")

        QMessageBox.information(
            self, "Export complete",
            "Saved files:\n" + "\n".join(f"  • {f}" for f in saved))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _metric_card(label: str, value: str, sub: str) -> QFrame:
        """Creates a summary metric card widget."""
        card = QFrame()
        card.setStyleSheet(
            "QFrame{background:#f0f4f8;border-radius:8px;padding:4px}")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(2)

        lbl = QLabel(label)
        lbl.setFont(QFont("Arial", 9))
        lbl.setStyleSheet("color:#666")
        layout.addWidget(lbl)

        val = QLabel(value)
        val.setFont(QFont("Arial", 20, QFont.Bold))
        val.setStyleSheet("color:#0C447C")
        layout.addWidget(val)

        sub_lbl = QLabel(sub)
        sub_lbl.setFont(QFont("Arial", 8))
        sub_lbl.setStyleSheet("color:#888")
        layout.addWidget(sub_lbl)

        return card

    @staticmethod
    def _btn_style(color: str, large: bool = False) -> str:
        h = "48px" if large else "36px"
        return (f"QPushButton{{background:{color};color:white;"
                f"border:none;border-radius:8px;"
                f"padding:6px 18px;min-height:{h};"
                f"font-size:{'13' if large else '11'}px}}"
                f"QPushButton:hover{{background:{color}cc}}"
                f"QPushButton:disabled{{background:#ccc;color:#888}}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Clean palette
    palette = QPalette()
    palette.setColor(QPalette.Window,      QColor("#f5f6fa"))
    palette.setColor(QPalette.WindowText,  QColor("#1a1a2e"))
    palette.setColor(QPalette.Base,        QColor("#ffffff"))
    palette.setColor(QPalette.AlternateBase, QColor("#f0f4f8"))
    palette.setColor(QPalette.Button,      QColor("#ffffff"))
    palette.setColor(QPalette.ButtonText,  QColor("#1a1a2e"))
    palette.setColor(QPalette.Highlight,   QColor("#185FA5"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    window = MABApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
