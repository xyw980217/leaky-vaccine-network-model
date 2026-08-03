#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Supplementary Figure S3:
coverage-dependent performance of the high-exposure and oldest-first strategies.

The script reads the retained simulation summary produced by
simulations/run_primary_vaccination_experiments.py. It can read either:
  1. Leaky2.zip directly;
  2. an extracted CSV.GZ file; or
  3. the legacy Excel export.

Outputs:
  - FigureS3.pdf
  - FigureS3.png
  - FigureS3_summary.csv

Main effect-size definition (matching the SI and main-text heatmaps):
    Delta_m(s) = 100 * (Y_random - Y_s) / Y_0
where Y_0 is the no-vaccination outcome for the same (VES_0, R0) setting.
"""

from __future__ import annotations

import gzip
import io
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_RESULTS_DIR = PROJECT_ROOT / 'results' / 'primary_simulation'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'figures' / 'supplementary'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR = PROJECT_ROOT

# The first existing path is used automatically.
INPUT_CANDIDATES = [
    PRIMARY_RESULTS_DIR / "Simulation_Results_All_Scenarios_With_Uncertainty.csv.gz",
    PRIMARY_RESULTS_DIR / "Simulation_Results_All_Scenarios_Parallel.xlsx",
    BASE_DIR / "Leaky2.zip",
]

ZIP_MEMBER = (
    "Leaky2/retained_simulation_data/"
    "Simulation_Results_All_Scenarios_With_Uncertainty.csv.gz"
)

OUTPUT_PDF = OUTPUT_DIR / "FigureS3.pdf"
OUTPUT_PNG = OUTPUT_DIR / "FigureS3.png"
OUTPUT_DATA = OUTPUT_DIR / "FigureS3_summary.csv"

STRATEGIES = {
    "High-exposure strategy": "High Strength (Node)",
    "Oldest-first strategy": "Priority: Oldest First",
}

OUTCOMES = {
    "Final infection proportion": "Infection Rate",
    "Epidemic peak size": "Peak Size",
    "Expected deaths": "Expected Deaths",
}

COVERAGE_TICKS = [5, 20, 35, 50, 65, 80, 95]


# -----------------------------------------------------------------------------
# Data loading and validation
# -----------------------------------------------------------------------------
def load_results() -> tuple[pd.DataFrame, Path]:
    """Load the combined retained simulation summary."""
    input_path = next((p for p in INPUT_CANDIDATES if p.exists()), None)
    if input_path is None:
        searched = "\n".join(f"  - {p}" for p in INPUT_CANDIDATES)
        raise FileNotFoundError(
            "No simulation-result file was found. Searched:\n" + searched
        )

    suffixes = input_path.suffixes

    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, "r") as zf:
            member = ZIP_MEMBER
            if member not in zf.namelist():
                matches = [
                    name for name in zf.namelist()
                    if name.endswith(
                        "Simulation_Results_All_Scenarios_With_Uncertainty.csv.gz"
                    )
                ]
                if len(matches) != 1:
                    raise FileNotFoundError(
                        "Could not identify the combined CSV.GZ inside the ZIP. "
                        f"Matches found: {matches}"
                    )
                member = matches[0]

            compressed_bytes = zf.read(member)
            with gzip.GzipFile(fileobj=io.BytesIO(compressed_bytes)) as gz:
                df = pd.read_csv(gz)

    elif suffixes[-2:] == [".csv", ".gz"]:
        df = pd.read_csv(input_path)

    elif input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)

    else:
        raise ValueError(f"Unsupported input format: {input_path}")

    required_columns = {
        "VES_0", "R0", "Coverage", "Strategy",
        "Infection Rate", "Peak Size", "Expected Deaths",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    numeric_columns = [
        "VES_0", "R0", "Coverage",
        "Infection Rate", "Peak Size", "Expected Deaths",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    # Avoid floating-point artefacts such as 0.499999999999.
    df["Coverage"] = df["Coverage"].round(4)

    duplicate_mask = df.duplicated(
        subset=["VES_0", "R0", "Coverage", "Strategy"], keep=False
    )
    if duplicate_mask.any():
        duplicated = df.loc[
            duplicate_mask,
            ["VES_0", "R0", "Coverage", "Strategy"]
        ].head(10)
        raise ValueError(
            "Duplicate scenario-summary rows were detected. Examples:\n"
            + duplicated.to_string(index=False)
        )

    return df, input_path


# -----------------------------------------------------------------------------
# Effect-size calculation
# -----------------------------------------------------------------------------
def calculate_additional_reductions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Delta_m(s) = 100 * (Y_random - Y_s) / Y_0.

    Y_0 is taken only from the Random, coverage=0 row because the supplied
    simulation code explicitly designates that row as the common baseline.
    """
    outcome_columns = list(OUTCOMES.values())

    baseline = df.loc[
        (df["Strategy"] == "Random") & np.isclose(df["Coverage"], 0.0),
        ["VES_0", "R0", *outcome_columns],
    ].copy()
    baseline = baseline.rename(
        columns={column: f"{column}__Y0" for column in outcome_columns}
    )

    random_at_coverage = df.loc[
        df["Strategy"] == "Random",
        ["VES_0", "R0", "Coverage", *outcome_columns],
    ].copy()
    random_at_coverage = random_at_coverage.rename(
        columns={column: f"{column}__Random" for column in outcome_columns}
    )

    expected_scenarios = (
        df[["VES_0", "R0"]].drop_duplicates().shape[0]
    )
    if expected_scenarios != 85:
        print(
            f"[WARNING] Expected 85 (VES_0, R0) combinations but found "
            f"{expected_scenarios}.",
            file=sys.stderr,
        )

    long_frames: list[pd.DataFrame] = []

    for display_name, internal_name in STRATEGIES.items():
        targeted = df.loc[
            (df["Strategy"] == internal_name) & (df["Coverage"] > 0),
            ["VES_0", "R0", "Coverage", *outcome_columns],
        ].copy()

        merged = (
            targeted
            .merge(baseline, on=["VES_0", "R0"], how="left", validate="many_to_one")
            .merge(
                random_at_coverage,
                on=["VES_0", "R0", "Coverage"],
                how="left",
                validate="one_to_one",
            )
        )

        if merged.isna().any().any():
            missing_cols = merged.columns[merged.isna().any()].tolist()
            raise ValueError(
                f"Missing baseline or random-comparison values after merging: "
                f"{missing_cols}"
            )

        for outcome_name, outcome_column in OUTCOMES.items():
            y0 = merged[f"{outcome_column}__Y0"].to_numpy(dtype=float)
            y_random = merged[f"{outcome_column}__Random"].to_numpy(dtype=float)
            y_target = merged[outcome_column].to_numpy(dtype=float)

            if np.any(np.isclose(y0, 0.0)):
                raise ZeroDivisionError(
                    f"A zero no-vaccination baseline was found for {outcome_name}."
                )

            delta = 100.0 * (y_random - y_target) / y0

            frame = merged[["VES_0", "R0", "Coverage"]].copy()
            frame["Strategy"] = display_name
            frame["Outcome"] = outcome_name
            frame["Additional reduction (%)"] = delta
            long_frames.append(frame)

    long_df = pd.concat(long_frames, ignore_index=True)

    summary = (
        long_df
        .groupby(["Strategy", "Outcome", "Coverage"], as_index=False)
        .agg(
            Mean=("Additional reduction (%)", "mean"),
            Median=("Additional reduction (%)", "median"),
            Q25=("Additional reduction (%)", lambda x: x.quantile(0.25)),
            Q75=("Additional reduction (%)", lambda x: x.quantile(0.75)),
            Q10=("Additional reduction (%)", lambda x: x.quantile(0.10)),
            Q90=("Additional reduction (%)", lambda x: x.quantile(0.90)),
            Positive_count=("Additional reduction (%)", lambda x: int((x > 0).sum())),
            Parameter_count=("Additional reduction (%)", "size"),
        )
        .sort_values(["Strategy", "Outcome", "Coverage"])
        .reset_index(drop=True)
    )

    # Figure S3 uses the mean because the SI text reports means across the 85
    # parameter combinations. The shaded band is the interquartile range across
    # parameter combinations; it is descriptive heterogeneity, not Monte Carlo CI.
    return summary


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def set_shared_column_limits(
    axes: np.ndarray,
    summary: pd.DataFrame,
    outcome_name: str,
    column_index: int,
) -> None:
    """Use the same y-axis range for both strategies within an outcome column."""
    subset = summary.loc[summary["Outcome"] == outcome_name]
    lower = float(min(subset["Q25"].min(), subset["Mean"].min(), 0.0))
    upper = float(max(subset["Q75"].max(), subset["Mean"].max(), 0.0))

    span = upper - lower
    if span <= 0:
        span = 1.0
    padding = 0.12 * span

    for row_index in range(axes.shape[0]):
        axes[row_index, column_index].set_ylim(lower - padding, upper + padding)


def plot_figure(summary: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    strategy_order = list(STRATEGIES.keys())
    outcome_order = list(OUTCOMES.keys())

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(12.0, 7.0),
        sharex=True,
    )

    panel_letters = iter("abcdef")
    first_line = None
    first_band = None

    for row_index, strategy_name in enumerate(strategy_order):
        for column_index, outcome_name in enumerate(outcome_order):
            ax = axes[row_index, column_index]
            panel = summary.loc[
                (summary["Strategy"] == strategy_name)
                & (summary["Outcome"] == outcome_name)
            ].sort_values("Coverage")

            x = panel["Coverage"].to_numpy(dtype=float) * 100.0
            mean = panel["Mean"].to_numpy(dtype=float)
            q25 = panel["Q25"].to_numpy(dtype=float)
            q75 = panel["Q75"].to_numpy(dtype=float)

            # Reference lines are deliberately drawn first and lightly so the
            # coverage-response curve remains visually dominant.
            ax.axhline(0.0, linestyle="--", linewidth=0.8, alpha=0.55, zorder=0)
            ax.axvline(50.0, linestyle=":", linewidth=0.8, alpha=0.55, zorder=0)

            line, = ax.plot(
                x,
                mean,
                marker="o",
                markersize=3.3,
                linewidth=1.6,
                label="Mean across parameter combinations",
                zorder=3,
            )
            band = ax.fill_between(
                x,
                q25,
                q75,
                color=line.get_color(),
                alpha=0.18,
                linewidth=0,
                label="Interquartile range",
                zorder=1,
            )

            if first_line is None:
                first_line = line
                first_band = band

            ax.set_xlim(3, 97)
            ax.set_xticks(COVERAGE_TICKS)
            ax.grid(axis="y", linewidth=0.5, alpha=0.28)
            ax.tick_params(direction="out", length=3, width=0.8)

            if row_index == 0:
                ax.set_title(outcome_name, pad=8)

            if row_index == 1:
                ax.set_xlabel("Vaccination coverage (%)")

            panel_letter = next(panel_letters)
            ax.text(
                -0.13,
                1.04,
                panel_letter,
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="bottom",
                ha="left",
            )

    for column_index, outcome_name in enumerate(outcome_order):
        set_shared_column_limits(axes, summary, outcome_name, column_index)

    # Strategy labels are placed outside the plotting area to avoid repeating
    # the strategy name in all three panels of a row.
    fig.text(
        0.018,
        0.705,
        "High-exposure strategy",
        rotation=90,
        va="center",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.018,
        0.285,
        "Oldest-first strategy",
        rotation=90,
        va="center",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.042,
        0.50,
        "Additional reduction relative to random strategy (%)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
    )

    if first_line is not None and first_band is not None:
        fig.legend(
            handles=[first_line, first_band],
            labels=[
                "Mean across 85 parameter combinations",
                "Interquartile range across parameter combinations",
            ],
            loc="upper center",
            bbox_to_anchor=(0.55, 0.985),
            ncol=2,
            frameon=False,
        )

    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.10,
        top=0.90,
        wspace=0.24,
        hspace=0.30,
    )

    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)


def print_cross_checks(summary: pd.DataFrame) -> None:
    """Print selected values that should reproduce the current SI text."""
    checks = [
        ("High-exposure strategy", "Epidemic peak size", 0.05),
        ("High-exposure strategy", "Epidemic peak size", 0.35),
        ("High-exposure strategy", "Epidemic peak size", 0.50),
        ("High-exposure strategy", "Final infection proportion", 0.40),
        ("High-exposure strategy", "Expected deaths", 0.40),
        ("Oldest-first strategy", "Expected deaths", 0.30),
        ("Oldest-first strategy", "Expected deaths", 0.50),
        ("Oldest-first strategy", "Epidemic peak size", 0.45),
        ("Oldest-first strategy", "Final infection proportion", 0.50),
    ]

    print("\n[Cross-checks against the current SI text]")
    for strategy, outcome, coverage in checks:
        row = summary.loc[
            (summary["Strategy"] == strategy)
            & (summary["Outcome"] == outcome)
            & np.isclose(summary["Coverage"], coverage)
        ]
        if row.empty:
            raise ValueError(
                f"Cross-check row not found: {strategy}, {outcome}, {coverage}"
            )
        value = row.iloc[0]
        print(
            f"  {strategy} | {outcome} | coverage={coverage:.0%}: "
            f"mean={value['Mean']:.2f}%, "
            f"positive={int(value['Positive_count'])}/"
            f"{int(value['Parameter_count'])}"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    df, input_path = load_results()
    print(f"[INFO] Loaded: {input_path}")
    print(f"[INFO] Rows: {len(df):,}")

    summary = calculate_additional_reductions(df)
    summary.to_csv(OUTPUT_DATA, index=False)

    print_cross_checks(summary)
    plot_figure(summary)

    print(f"\n[SAVED] {OUTPUT_PDF}")
    print(f"[SAVED] {OUTPUT_PNG}")
    print(f"[SAVED] {OUTPUT_DATA}")


if __name__ == "__main__":
    main()
