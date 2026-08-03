#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Supplementary Figure S5 for the oldest-first strategy.

The figure summarizes disease-parameter robustness at 50% vaccination
coverage and omits panel letters and outer borders while retaining cell
annotations, axes, and colour scales.
"""

from __future__ import annotations

import math
import zipfile
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS_RESULTS_DIR = PROJECT_ROOT / 'results' / 'disease_parameter_robustness'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'figures' / 'supplementary'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR = PROJECT_ROOT

ZIP_CANDIDATES = [
    ROBUSTNESS_RESULTS_DIR / "Robustness_disease.zip",
]

SUMMARY_CANDIDATES = [
    ROBUSTNESS_RESULTS_DIR / "Robustness_Additional_Reduction_Summary.xlsx",
]

OUTPUT_PDF = OUTPUT_DIR / "FigureS5.pdf"
OUTPUT_PNG = OUTPUT_DIR / "FigureS5.png"


# ============================================================
# 2. Figure settings
# ============================================================

SCENARIO_ORDER = ["COVID_like", "Influenza_like", "RSV_like"]
SCENARIO_LABELS = {
    "COVID_like": "COVID-like",
    "Influenza_like": "Influenza-like",
    "RSV_like": "RSV-like",
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

TARGET_COVERAGE = 0.50
CMAP = plt.get_cmap("RdYlGn")
ANNOTATE_CELLS = True
ANNOTATION_DECIMALS = 2
FIGSIZE = (12.8, 9.2)
DPI = 600


# ============================================================
# 3. Helpers
# ============================================================

def _normalise_name(value: object) -> str:
    text = str(value).strip()
    text = text.replace("-", "_").replace(" ", "_")
    text = text.replace("/", "_")
    text = text.replace("(", "").replace(")", "")
    text = text.replace(":", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.lower()


def _standardise_scenario(value: object) -> str:
    key = _normalise_name(value)
    aliases = {
        "covid_like": "COVID_like",
        "covidlike": "COVID_like",
        "influenza_like": "Influenza_like",
        "influenzalike": "Influenza_like",
        "rsv_like": "RSV_like",
        "rsvlike": "RSV_like",
    }
    if key not in aliases:
        raise ValueError(f"Unrecognised disease-like scenario: {value!r}")
    return aliases[key]


def _standardise_outcome(value: object) -> str:
    key = _normalise_name(value)
    aliases = {
        "final_infection_proportion": "Final Infection Proportion",
        "infection_rate": "Final Infection Proportion",
        "final_infection_rate": "Final Infection Proportion",
        "epidemic_peak_size": "Epidemic Peak Size",
        "peak_size": "Epidemic Peak Size",
        "expected_deaths": "Expected Deaths",
        "expected_death": "Expected Deaths",
    }
    if key not in aliases:
        raise ValueError(f"Unrecognised outcome: {value!r}")
    return aliases[key]


def _is_oldest_first_strategy(value: object) -> bool:
    key = _normalise_name(value)

    exact_matches = {
        "priority_oldest_first",
        "priority_oldest_first_",
        "oldest_first",
        "oldest_first_strategy",
    }
    if key in exact_matches:
        return True

    return ("oldest" in key and "first" in key)


def _nice_symmetric_limit(max_abs: float) -> float:
    if not np.isfinite(max_abs) or max_abs <= 0:
        return 1.0

    if max_abs <= 1:
        step = 0.25
    elif max_abs <= 2:
        step = 0.5
    elif max_abs <= 5:
        step = 1.0
    elif max_abs <= 10:
        step = 2.0
    elif max_abs <= 20:
        step = 5.0
    else:
        magnitude = 10 ** math.floor(math.log10(max_abs))
        step = magnitude / 2.0

    return float(math.ceil(max_abs / step) * step)


def _text_colour(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, _ = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.60 else "white"


# ============================================================
# 4. Data loading
# ============================================================

def load_summary_table() -> tuple[pd.DataFrame, str]:
    for path in SUMMARY_CANDIDATES:
        if path.exists():
            df = pd.read_excel(path, sheet_name="Additional_Reduction")
            return df, str(path)

    for zip_path in ZIP_CANDIDATES:
        if not zip_path.exists():
            continue

        with zipfile.ZipFile(zip_path, "r") as archive:
            matches = [
                name for name in archive.namelist()
                if name.endswith("Robustness_Additional_Reduction_Summary.xlsx")
            ]
            if not matches:
                continue

            workbook_bytes = archive.read(matches[0])
            df = pd.read_excel(
                BytesIO(workbook_bytes),
                sheet_name="Additional_Reduction",
            )
            return df, f"{zip_path}!/{matches[0]}"

    raise FileNotFoundError(
        "Could not find Robustness_Additional_Reduction_Summary.xlsx "
        "or Robustness_disease.zip beside the plotting script."
    )


def prepare_summary_table(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[float], list[float]]:
    required = {
        "Scenario",
        "VES_0",
        "R0",
        "Coverage",
        "Strategy",
        "Outcome",
        "Additional Reduction (%)",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    out = df.copy()
    out["Scenario"] = out["Scenario"].map(_standardise_scenario)
    out["Outcome"] = out["Outcome"].map(_standardise_outcome)
    out["VES_0"] = pd.to_numeric(out["VES_0"], errors="coerce")
    out["R0"] = pd.to_numeric(out["R0"], errors="coerce")
    out["Coverage"] = pd.to_numeric(out["Coverage"], errors="coerce")
    out["Additional Reduction (%)"] = pd.to_numeric(
        out["Additional Reduction (%)"],
        errors="coerce",
    )

    out = out[
        np.isclose(out["Coverage"], TARGET_COVERAGE, atol=0.01)
        & out["Strategy"].map(_is_oldest_first_strategy)
        & out["Scenario"].isin(SCENARIO_ORDER)
        & out["Outcome"].isin(OUTCOME_ORDER)
    ].copy()

    r0_order = sorted(out["R0"].dropna().unique().astype(float).tolist())
    ves_order = sorted(
        out["VES_0"].dropna().unique().astype(float).tolist(),
        reverse=True,
    )

    out = (
        out.groupby(
            ["Scenario", "Outcome", "VES_0", "R0"],
            as_index=False,
        )["Additional Reduction (%)"]
        .mean()
    )

    expected_rows = (
        len(SCENARIO_ORDER)
        * len(OUTCOME_ORDER)
        * len(ves_order)
        * len(r0_order)
    )

    if len(out) != expected_rows:
        counts = (
            out.groupby(["Scenario", "Outcome"])
            .size()
            .unstack(fill_value=0)
        )
        raise ValueError(
            f"Expected {expected_rows} plotted values but found {len(out)}.\n"
            f"Detected R0 values: {r0_order}\n"
            f"Detected VE_S,0 values: {ves_order}\n"
            f"Counts by scenario and outcome:\n{counts}\n"
            f"Please check the strategy label for the oldest-first strategy."
        )

    print(f"[INFO] Detected R0 grid: {r0_order}")
    print(f"[INFO] Detected VE_S,0 grid: {ves_order}")
    print(f"[INFO] Oldest-first plotted values: {len(out)}")

    return out, r0_order, ves_order


# ============================================================
# 5. Plotting
# ============================================================

def panel_matrix(
    plot_df: pd.DataFrame,
    outcome: str,
    scenario: str,
    r0_order: list[float],
    ves_order: list[float],
) -> np.ndarray:
    subset = plot_df[
        (plot_df["Outcome"] == outcome)
        & (plot_df["Scenario"] == scenario)
    ]

    pivot = subset.pivot(
        index="VES_0",
        columns="R0",
        values="Additional Reduction (%)",
    ).reindex(index=ves_order, columns=r0_order)

    if pivot.isna().any().any():
        raise ValueError(
            f"Missing values for outcome={outcome}, scenario={scenario}."
        )

    return pivot.to_numpy(dtype=float)


def plot_figure(
    plot_df: pd.DataFrame,
    r0_order: list[float],
    ves_order: list[float],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=FIGSIZE)
    grid = GridSpec(
        nrows=3,
        ncols=4,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 0.045],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.18,
        hspace=0.28,
    )

    axes = np.empty((3, 3), dtype=object)

    limits: dict[str, float] = {}
    for outcome in OUTCOME_ORDER:
        values = plot_df.loc[
            plot_df["Outcome"] == outcome,
            "Additional Reduction (%)",
        ].to_numpy(dtype=float)
        limits[outcome] = _nice_symmetric_limit(
            np.nanmax(np.abs(values))
        )

    for row_idx, outcome in enumerate(OUTCOME_ORDER):
        limit = limits[outcome]
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        last_image = None

        for col_idx, scenario in enumerate(SCENARIO_ORDER):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            axes[row_idx, col_idx] = ax

            values = panel_matrix(
                plot_df,
                outcome,
                scenario,
                r0_order,
                ves_order,
            )

            last_image = ax.imshow(
                values,
                cmap=CMAP,
                norm=norm,
                interpolation="nearest",
                aspect="auto",
                origin="upper",
            )

            # Retain white separators between heatmap cells.
            ax.set_xticks(
                np.arange(-0.5, len(r0_order), 1),
                minor=True,
            )
            ax.set_yticks(
                np.arange(-0.5, len(ves_order), 1),
                minor=True,
            )
            ax.grid(which="minor", color="white", linewidth=1.0)
            ax.tick_params(which="minor", bottom=False, left=False)

            ax.set_xticks(np.arange(len(r0_order)))
            ax.set_yticks(np.arange(len(ves_order)))

            if row_idx == len(OUTCOME_ORDER) - 1:
                ax.set_xticklabels([f"{value:g}" for value in r0_order])
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)

            if col_idx == 0:
                ax.set_yticklabels([f"{value:.1f}" for value in ves_order])
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)

            if row_idx == 0:
                ax.set_title(SCENARIO_LABELS[scenario], pad=8)

            if ANNOTATE_CELLS:
                for i in range(values.shape[0]):
                    for j in range(values.shape[1]):
                        value = float(values[i, j])
                        rgba = CMAP(norm(value))
                        ax.text(
                            j,
                            i,
                            f"{value:.{ANNOTATION_DECIMALS}f}",
                            ha="center",
                            va="center",
                            fontsize=7.2,
                            color=_text_colour(rgba),
                        )

            # Hide subplot borders.
            for spine in ax.spines.values():
                spine.set_visible(False)

        cax = fig.add_subplot(grid[row_idx, 3])
        cbar = fig.colorbar(last_image, cax=cax)
        cbar.set_ticks(np.linspace(-limit, limit, 5))
        cbar.ax.tick_params(labelsize=8.0, length=2.5)

        cbar.outline.set_visible(False)
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)

    for row_idx, outcome in enumerate(OUTCOME_ORDER):
        position = axes[row_idx, 0].get_position()
        fig.text(
            0.025,
            0.5 * (position.y0 + position.y1),
            OUTCOME_LABELS[outcome],
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )

    fig.text(
        0.50,
        0.035,
        r"Basic reproduction number ($R_0$)",
        ha="center",
        va="center",
        fontsize=10.5,
    )
    fig.text(
        0.075,
        0.50,
        r"Baseline vaccine efficacy against susceptibility ($VE_{S,0}$)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10.5,
    )
    # fig.text(
    #     0.985,
    #     0.50,
    #     "Additional reduction relative to random strategy (%)",
    #     rotation=90,
    #     ha="center",
    #     va="center",
    #     fontsize=10.0,
    # )

    fig.subplots_adjust(
        left=0.13,
        right=0.955,
        bottom=0.09,
        top=0.95,
    )

    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    raw, source = load_summary_table()
    plot_df, r0_order, ves_order = prepare_summary_table(raw)
    plot_figure(plot_df, r0_order, ves_order)

    print(f"[INFO] Loaded: {source}")
    print(f"[SAVED] {OUTPUT_PDF}")
    print(f"[SAVED] {OUTPUT_PNG}")


if __name__ == "__main__":
    main()