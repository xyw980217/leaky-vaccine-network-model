#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Figure S6
Network-structure robustness of vaccination-strategy performance.

Figure layout:
    2 rows x 3 columns

Rows:
    High-exposure strategy
    Oldest-first strategy

Columns:
    Final infection proportion
    Epidemic peak size
    Expected deaths

For each network type, the 12 points correspond to the 12 epidemiological
parameter combinations:
    4 R0 values x 3 baseline VE_S,0 values.

The input sheet "Parameter_Level_Reduction" has already averaged additional
reduction across stochastic network replicates for each R0 x VE_S,0 setting.
The plotting code therefore treats parameter combinations, rather than
individual network realizations, as the plotted observations.

The figure shows:
    - jittered points: individual R0 x VE_S,0 combinations;
    - box plots: median and interquartile range across the 12 combinations;
    - diamonds: mean across the 12 combinations;
    - horizontal dashed line: zero additional reduction;
    - vertical separator: survey-derived reference/local structural
      perturbations versus idealized topology controls.

Input:
    Network_Robustness_Additional_Reduction_Summary.xlsx
or:
    robustness_new.zip containing that workbook.

Outputs:
    FigureS6_Network_Structure_Robustness.pdf
    FigureS6_Network_Structure_Robustness.png
    FigureS6_Network_Structure_Robustness_plot_data.csv
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS_RESULTS_DIR = PROJECT_ROOT / 'results' / 'network_structure_robustness'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'figures' / 'supplementary'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR = PROJECT_ROOT

INPUT_XLSX_CANDIDATES = [
    ROBUSTNESS_RESULTS_DIR / "Network_Robustness_Additional_Reduction_Summary.xlsx",
]

INPUT_ZIP_CANDIDATES = [
    ROBUSTNESS_RESULTS_DIR / "robustness_new.zip",
]

OUTPUT_PDF = OUTPUT_DIR / "FigureS6.pdf"
OUTPUT_PNG = OUTPUT_DIR / "FigureS6.png"
OUTPUT_DATA = OUTPUT_DIR / "FigureS6_Network_Structure_Robustness_plot_data.csv"


# ============================================================
# 2. Plot settings
# ============================================================

NETWORK_ORDER = [
    "Empirical",
    "CommunityResampled",
    "SmallWorldCommunity",
    "HouseholdAttenuatedCompensated",
    "ERRandom",
    "BAScaleFree",
    "CompletelyRandomized",
]

NETWORK_LABELS = {
    # The data key remains "Empirical" because it is the value stored in the
    # robustness workbook. Only the figure-facing label is changed to match
    # the terminology used in the Supplementary Information.
    "Empirical": "Survey-derived\nreference",
    "CommunityResampled": "Community-\nresampled",
    "SmallWorldCommunity": "Small-world\ncommunity",
    "HouseholdAttenuatedCompensated": "Household-\nattenuated",
    "ERRandom": "Erdős–Rényi\nrandom",
    "BAScaleFree": "Barabási–Albert\nscale-free",
    "CompletelyRandomized": "Fully\nrandomized",
}

STRATEGY_ORDER = [
    "High Strength (Node)",
    "Priority: Oldest First",
]

STRATEGY_LABELS = {
    "High Strength (Node)": "High-exposure strategy",
    "Priority: Oldest First": "Oldest-first strategy",
}

OUTCOME_ORDER = [
    "Final Infection Proportion",
    "Epidemic Peak Size",
    "Expected Deaths",
]

OUTCOME_LABELS = {
    "Final Infection Proportion": "Final infection proportion",
    "Epidemic Peak Size": "Epidemic peak size",
    "Expected Deaths": "Expected deaths",
}

# Colours used in the preceding manuscript figures.
STRATEGY_COLOURS = {
    "High Strength (Node)": (110 / 255, 151 / 255, 177 / 255),
    "Priority: Oldest First": (178 / 255, 54 / 255, 36 / 255),
}

FIGSIZE = (15.8, 8.6)
DPI = 600
SHOW_PANEL_LETTERS = False

# Fixed random seed used only for visual jitter.
JITTER_SEED = 20260726


# ============================================================
# 3. Data loading
# ============================================================

def load_parameter_level_results() -> tuple[pd.DataFrame, str]:
    """Read the Parameter_Level_Reduction sheet from XLSX or ZIP."""
    for path in INPUT_XLSX_CANDIDATES:
        if path.exists():
            data = pd.read_excel(
                path,
                sheet_name="Parameter_Level_Reduction",
            )
            return data, str(path)

    for zip_path in INPUT_ZIP_CANDIDATES:
        if not zip_path.exists():
            continue

        with zipfile.ZipFile(zip_path, "r") as archive:
            matches = [
                name for name in archive.namelist()
                if name.endswith(
                    "Network_Robustness_Additional_Reduction_Summary.xlsx"
                )
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    "Expected exactly one "
                    "Network_Robustness_Additional_Reduction_Summary.xlsx "
                    f"inside {zip_path.name}, but found: {matches}"
                )

            workbook_bytes = archive.read(matches[0])
            data = pd.read_excel(
                BytesIO(workbook_bytes),
                sheet_name="Parameter_Level_Reduction",
            )
            return data, f"{zip_path}!/{matches[0]}"

    raise FileNotFoundError(
        "Could not find Network_Robustness_Additional_Reduction_Summary.xlsx "
        "or robustness_new.zip beside the plotting script."
    )


def prepare_plot_data(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "Network_Type",
        "VES_0",
        "R0",
        "Strategy",
        "Outcome",
        "Mean_additional_reduction_across_network_replicates",
        "SD_across_network_replicates",
        "N_network_replicates",
    }
    missing = required_columns.difference(data.columns)
    if missing:
        raise KeyError(
            "The Parameter_Level_Reduction sheet is missing columns: "
            + ", ".join(sorted(missing))
        )

    plot_data = data.copy()

    numeric_columns = [
        "VES_0",
        "R0",
        "Mean_additional_reduction_across_network_replicates",
        "SD_across_network_replicates",
        "N_network_replicates",
    ]
    for column in numeric_columns:
        plot_data[column] = pd.to_numeric(
            plot_data[column],
            errors="coerce",
        )

    plot_data = plot_data[
        plot_data["Network_Type"].isin(NETWORK_ORDER)
        & plot_data["Strategy"].isin(STRATEGY_ORDER)
        & plot_data["Outcome"].isin(OUTCOME_ORDER)
    ].copy()

    plot_data = (
        plot_data.groupby(
            ["Network_Type", "VES_0", "R0", "Strategy", "Outcome"],
            as_index=False,
        )
        .agg(
            Additional_reduction=(
                "Mean_additional_reduction_across_network_replicates",
                "mean",
            ),
            SD_across_network_replicates=(
                "SD_across_network_replicates",
                "mean",
            ),
            N_network_replicates=(
                "N_network_replicates",
                "max",
            ),
        )
    )

    expected_parameter_combinations = 12
    expected_total_rows = (
        len(NETWORK_ORDER)
        * len(STRATEGY_ORDER)
        * len(OUTCOME_ORDER)
        * expected_parameter_combinations
    )

    if len(plot_data) != expected_total_rows:
        counts = (
            plot_data.groupby(
                ["Network_Type", "Strategy", "Outcome"]
            )
            .size()
            .rename("N")
        )
        raise ValueError(
            f"Expected {expected_total_rows} parameter-level rows but found "
            f"{len(plot_data)}.\nCounts by panel:\n{counts}"
        )

    panel_counts = (
        plot_data.groupby(["Network_Type", "Strategy", "Outcome"])
        .size()
    )
    if not (panel_counts == expected_parameter_combinations).all():
        raise ValueError(
            "Each network-strategy-outcome panel should contain exactly "
            f"{expected_parameter_combinations} R0 x VE_S,0 combinations."
        )

    plot_data["Network_Type"] = pd.Categorical(
        plot_data["Network_Type"],
        categories=NETWORK_ORDER,
        ordered=True,
    )
    plot_data["Strategy"] = pd.Categorical(
        plot_data["Strategy"],
        categories=STRATEGY_ORDER,
        ordered=True,
    )
    plot_data["Outcome"] = pd.Categorical(
        plot_data["Outcome"],
        categories=OUTCOME_ORDER,
        ordered=True,
    )

    return plot_data.sort_values(
        ["Strategy", "Outcome", "Network_Type", "VES_0", "R0"]
    ).reset_index(drop=True)


# ============================================================
# 4. Plot helpers
# ============================================================

def calculate_outcome_limits(
    plot_data: pd.DataFrame,
    outcome: str,
) -> tuple[float, float]:
    """Use one shared y-axis range for both strategies within an outcome."""
    values = plot_data.loc[
        plot_data["Outcome"] == outcome,
        "Additional_reduction",
    ].to_numpy(dtype=float)

    lower = min(float(np.nanmin(values)), 0.0)
    upper = max(float(np.nanmax(values)), 0.0)
    span = upper - lower

    if span <= 0:
        span = 1.0

    padding = 0.10 * span
    return lower - padding, upper + padding


def fixed_jitter(number_of_points: int) -> np.ndarray:
    """
    Return deterministic, visually balanced horizontal jitter.

    The values are randomized reproducibly and then centred so that repeated
    execution produces the same figure.
    """
    rng = np.random.default_rng(JITTER_SEED + number_of_points)
    jitter = rng.uniform(-0.18, 0.18, size=number_of_points)
    jitter -= jitter.mean()
    return jitter


def style_axis(ax: plt.Axes) -> None:
    """Clean publication-style axes without a full rectangular frame."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.8,
    )

    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=0.5,
        alpha=0.22,
        zorder=0,
    )


# ============================================================
# 5. Main figure
# ============================================================

def plot_network_robustness(plot_data: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=FIGSIZE,
        sharex=True,
    )

    panel_letters = iter("abcdef")
    x_positions = np.arange(len(NETWORK_ORDER), dtype=float)

    for row_index, strategy in enumerate(STRATEGY_ORDER):
        strategy_colour = STRATEGY_COLOURS[strategy]

        for column_index, outcome in enumerate(OUTCOME_ORDER):
            ax = axes[row_index, column_index]
            panel = plot_data[
                (plot_data["Strategy"] == strategy)
                & (plot_data["Outcome"] == outcome)
            ].copy()

            values_by_network: list[np.ndarray] = []

            for network in NETWORK_ORDER:
                values = (
                    panel.loc[
                        panel["Network_Type"] == network,
                        "Additional_reduction",
                    ]
                    .to_numpy(dtype=float)
                )
                values_by_network.append(values)

            boxplot = ax.boxplot(
                values_by_network,
                positions=x_positions,
                widths=0.52,
                patch_artist=True,
                showfliers=False,
                whis=1.5,
                medianprops={
                    "color": "black",
                    "linewidth": 1.2,
                },
                boxprops={
                    "edgecolor": strategy_colour,
                    "linewidth": 1.1,
                },
                whiskerprops={
                    "color": strategy_colour,
                    "linewidth": 0.9,
                },
                capprops={
                    "color": strategy_colour,
                    "linewidth": 0.9,
                },
            )

            for patch in boxplot["boxes"]:
                patch.set_facecolor(strategy_colour)
                patch.set_alpha(0.22)

            for network_index, values in enumerate(values_by_network):
                jitter = fixed_jitter(len(values))
                ax.scatter(
                    np.full(len(values), x_positions[network_index]) + jitter,
                    values,
                    s=18,
                    facecolor=strategy_colour,
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.82,
                    zorder=3,
                )

                mean_value = float(np.mean(values))
                ax.scatter(
                    x_positions[network_index],
                    mean_value,
                    marker="D",
                    s=35,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=4,
                )

            ax.axhline(
                0.0,
                linestyle="--",
                linewidth=0.9,
                color="black",
                alpha=0.60,
                zorder=1,
            )

            # Separate the survey-derived reference/local perturbations from idealized controls.
            ax.axvline(
                3.5,
                linestyle=":",
                linewidth=0.9,
                color="black",
                alpha=0.55,
                zorder=1,
            )

            ymin, ymax = calculate_outcome_limits(plot_data, outcome)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(-0.55, len(NETWORK_ORDER) - 0.45)

            if row_index == 0:
                ax.set_title(OUTCOME_LABELS[outcome], pad=12)

            if row_index == 1:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(
                    [NETWORK_LABELS[network] for network in NETWORK_ORDER],
                    rotation=32,
                    ha="right",
                    rotation_mode="anchor",
                )
            else:
                ax.set_xticks(x_positions)
                ax.set_xticklabels([])

            if SHOW_PANEL_LETTERS:
                ax.text(
                    -0.10,
                    1.03,
                    next(panel_letters),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                next(panel_letters)

            style_axis(ax)

    # Strategy labels.
    fig.text(
        0.024,
        0.705,
        STRATEGY_LABELS[STRATEGY_ORDER[0]],
        rotation=90,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.024,
        0.285,
        STRATEGY_LABELS[STRATEGY_ORDER[1]],
        rotation=90,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )

    fig.text(
        0.050,
        0.50,
        "Additional reduction relative to random strategy (%)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10.0,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="0.45",
            markeredgecolor="white",
            markersize=6,
            label=r"One $R_0$--$VE_{S,0}$ combination",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="Mean across 12 combinations",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=2,
        frameon=False,
        fontsize=8.8,
    )

    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.19,
        top=0.88,
        wspace=0.22,
        hspace=0.26,
    )

    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    raw_data, source = load_parameter_level_results()
    plot_data = prepare_plot_data(raw_data)

    plot_data.to_csv(OUTPUT_DATA, index=False)
    plot_network_robustness(plot_data)

    print(f"[INFO] Loaded: {source}")
    print(
        "[INFO] Plotted parameter-level rows: "
        f"{len(plot_data)}"
    )
    print(f"[SAVED] {OUTPUT_PDF}")
    print(f"[SAVED] {OUTPUT_PNG}")
    print(f"[SAVED] {OUTPUT_DATA}")


if __name__ == "__main__":
    main()
