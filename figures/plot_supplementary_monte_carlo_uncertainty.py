#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Figures S7 and S8
Monte Carlo uncertainty and numerical stability.

Data sources
------------
1. Simulation_Results_All_Scenarios_With_Uncertainty.csv.gz
   Used for:
   - panels a-c: 95% Monte Carlo interval half-widths for the three raw outcomes;
   - panel d: direction classification of targeted-strategy differences across
     the 85 R0 x VE_S,0 combinations at 50% vaccination coverage.

2. selected_uncertainty_replicates/Selected_Replicate_Outcomes_*.csv.gz
   Used for:
   - Supplementary Figure S8: forest plot for five representative strategy
     comparisons.

The script can read these files directly from Leaky2(2).zip, Leaky2.zip, or
from the primary simulation results directory.

Figure layout
-------------
a-c. Distribution of 95% Monte Carlo interval half-widths at 50% vaccination
     coverage, separated by vaccination strategy.
d-e. Number of parameter combinations for which the approximate interval for
     additional reduction lies entirely below zero, includes zero, or lies
     entirely above zero, shown separately for the high-exposure and
     oldest-first strategies.
S8.  Additional-reduction estimates and approximate 95% Monte Carlo intervals
     for five representative settings retained at replicate level.

Statistical convention
----------------------
The simulations for different strategies are not paired. For the strategy
difference, the standard error is therefore propagated as:

    SE(Y_random - Y_strategy)
        = sqrt(SE_random^2 + SE_strategy^2)

The additional-reduction interval is calculated as:

    Delta = 100 * (Y_random - Y_strategy) / Y0

    SE(Delta) = 100 * SE(Y_random - Y_strategy) / Y0

matching the convention used in the current Supplementary Information. Y0 is
treated as the normalization-baseline point estimate; uncertainty in the
denominator is not propagated.

Outputs
-------
FigureS7a_Final_Infection_Uncertainty.pdf/png
FigureS7b_Peak_Size_Uncertainty.pdf/png
FigureS7c_Expected_Deaths_Uncertainty.pdf/png
FigureS7d_High_Exposure_Direction_Classification.pdf/png
FigureS7e_Oldest_First_Direction_Classification.pdf/png
FigureS8.pdf/png
FigureS7_raw_half_widths.csv
FigureS7_direction_classification.csv
FigureS8_representative_forest_data.csv
"""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths and output settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_RESULTS_DIR = PROJECT_ROOT / 'results' / 'primary_simulation'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'figures' / 'supplementary'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR = PROJECT_ROOT

ZIP_CANDIDATES = [
    BASE_DIR / "Leaky2(2).zip",
    BASE_DIR / "Leaky2.zip",
]

SUMMARY_FILE_CANDIDATES = [
    PRIMARY_RESULTS_DIR / "Simulation_Results_All_Scenarios_With_Uncertainty.csv.gz",
]

REPLICATE_DIR_CANDIDATES = [
    PRIMARY_RESULTS_DIR / "selected_uncertainty_replicates",
    PRIMARY_RESULTS_DIR / "s9_selected_replicates",
]

PANEL_OUTPUTS = {
    "a": {
        "pdf": OUTPUT_DIR / "FigureS7a_Final_Infection_Uncertainty.pdf",
        "png": OUTPUT_DIR / "FigureS7a_Final_Infection_Uncertainty.png",
    },
    "b": {
        "pdf": OUTPUT_DIR / "FigureS7b_Peak_Size_Uncertainty.pdf",
        "png": OUTPUT_DIR / "FigureS7b_Peak_Size_Uncertainty.png",
    },
    "c": {
        "pdf": OUTPUT_DIR / "FigureS7c_Expected_Deaths_Uncertainty.pdf",
        "png": OUTPUT_DIR / "FigureS7c_Expected_Deaths_Uncertainty.png",
    },
    "d": {
        "pdf": OUTPUT_DIR / "FigureS7d_High_Exposure_Direction_Classification.pdf",
        "png": OUTPUT_DIR / "FigureS7d_High_Exposure_Direction_Classification.png",
    },
    "e": {
        "pdf": OUTPUT_DIR / "FigureS7e_Oldest_First_Direction_Classification.pdf",
        "png": OUTPUT_DIR / "FigureS7e_Oldest_First_Direction_Classification.png",
    },
    "s8": {
        "pdf": OUTPUT_DIR / "FigureS8.pdf",
        "png": OUTPUT_DIR / "FigureS8.png",
    },
}

OUTPUT_HALF_WIDTH_DATA = OUTPUT_DIR / "FigureS7_raw_half_widths.csv"
OUTPUT_DIRECTION_DATA = OUTPUT_DIR / "FigureS7_direction_classification.csv"
OUTPUT_FOREST_DATA = OUTPUT_DIR / "FigureS8_representative_forest_data.csv"

DPI = 600
FIGSIZE = (15.5, 9.8)
SHOW_PANEL_LETTERS = False


# ============================================================
# 2. Analysis settings
# ============================================================

TARGET_COVERAGE = 0.50

STRATEGY_ORDER = [
    "Random",
    "High Strength (Node)",
    "Priority: Oldest First",
]

STRATEGY_LABELS = {
    "Random": "Random",
    "High Strength (Node)": "High-exposure",
    "Priority: Oldest First": "Oldest-first",
}

TARGETED_STRATEGIES = {
    "High-exposure strategy": "High Strength (Node)",
    "Oldest-first strategy": "Priority: Oldest First",
}

OUTCOMES = {
    "Final infection proportion": "Infection Rate",
    "Epidemic peak size": "Peak Size",
    "Expected deaths": "Expected Deaths",
}

HALF_WIDTH_AXIS_LABELS = {
    "Final infection proportion":
        "95% interval half-width\n(percentage points)",
    "Epidemic peak size":
        "95% interval half-width\n(infectious individuals)",
    "Expected deaths":
        "95% interval half-width\n(expected deaths)",
}

DIRECTION_ORDER = [
    "Entirely below zero",
    "Includes zero",
    "Entirely above zero",
]

REPRESENTATIVE_CASES = [
    {
        "Scenario Label": "HighExposure_Peak_StrongBenefit",
        "Strategy": "High Strength (Node)",
        "Outcome": "Peak Size",
        "Display Label":
            r"High-exposure: peak size"
            "\n"
            r"$R_0=2.0,\ VE_{S,0}=0.5$",
    },
    {
        "Scenario Label": "HighExposure_FinalInfection_Reversal",
        "Strategy": "High Strength (Node)",
        "Outcome": "Infection Rate",
        "Display Label":
            r"High-exposure: final infection proportion"
            "\n"
            r"$R_0=4.0,\ VE_{S,0}=0.8$",
    },
    {
        "Scenario Label": "OldestFirst_Deaths_StrongBenefit",
        "Strategy": "Priority: Oldest First",
        "Outcome": "Expected Deaths",
        "Display Label":
            r"Oldest-first: expected deaths"
            "\n"
            r"$R_0=1.2,\ VE_{S,0}=0.8$",
    },
    {
        "Scenario Label": "OldestFirst_Peak_StrongDisadvantage",
        "Strategy": "Priority: Oldest First",
        "Outcome": "Peak Size",
        "Display Label":
            r"Oldest-first: peak size"
            "\n"
            r"$R_0=1.5,\ VE_{S,0}=0.8$",
    },
    {
        "Scenario Label": "HighExposure_Peak_NearZero",
        "Strategy": "High Strength (Node)",
        "Outcome": "Peak Size",
        "Display Label":
            r"High-exposure: peak size"
            "\n"
            r"$R_0=18.6,\ VE_{S,0}=0.5$",
    },
]


# ============================================================
# 3. File loading
# ============================================================

def locate_zip_member(
    archive: zipfile.ZipFile,
    suffix: str,
) -> str:
    matches = [
        name for name in archive.namelist()
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one ZIP member ending with {suffix!r}, "
            f"but found {matches}"
        )
    return matches[0]


def load_main_summary() -> tuple[pd.DataFrame, str]:
    """Load the all-scenario summary from an extracted file or ZIP."""
    for path in SUMMARY_FILE_CANDIDATES:
        if path.exists():
            return pd.read_csv(path), str(path)

    for zip_path in ZIP_CANDIDATES:
        if not zip_path.exists():
            continue

        with zipfile.ZipFile(zip_path, "r") as archive:
            member = locate_zip_member(
                archive,
                "Simulation_Results_All_Scenarios_With_Uncertainty.csv.gz",
            )
            compressed_bytes = archive.read(member)
            with gzip.GzipFile(
                fileobj=io.BytesIO(compressed_bytes)
            ) as stream:
                data = pd.read_csv(stream)

            return data, f"{zip_path}!/{member}"

    raise FileNotFoundError(
        "Could not find the all-scenario uncertainty summary."
    )


def load_selected_replicates() -> tuple[pd.DataFrame, str]:
    """Load and combine the five pre-specified replicate-level uncertainty files."""
    extracted_files: list[Path] = []
    for directory in REPLICATE_DIR_CANDIDATES:
        if directory.exists():
            extracted_files.extend(
                sorted(directory.glob("Selected_Replicate_Outcomes_*.csv.gz"))
                + sorted(directory.glob("S9_Replicate_Outcomes_*.csv.gz"))
            )

    if extracted_files:
        frames = [pd.read_csv(path) for path in extracted_files]
        return (
            pd.concat(frames, ignore_index=True),
            str(extracted_files[0].parent),
        )

    for zip_path in ZIP_CANDIDATES:
        if not zip_path.exists():
            continue

        with zipfile.ZipFile(zip_path, "r") as archive:
            members = sorted(
                name for name in archive.namelist()
                if (
                    ("/selected_uncertainty_replicates/" in name
                     or "/s9_selected_replicates/" in name)
                    and name.endswith(".csv.gz")
                )
            )
            if not members:
                continue

            frames = []
            for member in members:
                compressed_bytes = archive.read(member)
                with gzip.GzipFile(
                    fileobj=io.BytesIO(compressed_bytes)
                ) as stream:
                    frames.append(pd.read_csv(stream))

            return (
                pd.concat(frames, ignore_index=True),
                f"{zip_path}!/selected replicate outcomes",
            )

    raise FileNotFoundError(
        "Could not find the selected replicate-level uncertainty outcomes."
    )


# ============================================================
# 4. Validation and summary calculations
# ============================================================

def validate_main_summary(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "VES_0",
        "R0",
        "Coverage",
        "Strategy",
    }

    for column in OUTCOMES.values():
        required.update(
            {
                column,
                f"{column} SD",
                f"{column} SE",
                f"{column} CI95 Lower",
                f"{column} CI95 Upper",
            }
        )

    missing = required.difference(data.columns)
    if missing:
        raise KeyError(
            "Main summary is missing required columns: "
            + ", ".join(sorted(missing))
        )

    work = data.copy()

    numeric_columns = [
        "VES_0",
        "R0",
        "Coverage",
    ]
    for column in OUTCOMES.values():
        numeric_columns.extend(
            [
                column,
                f"{column} SD",
                f"{column} SE",
                f"{column} CI95 Lower",
                f"{column} CI95 Upper",
            ]
        )

    for column in numeric_columns:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    return work


def make_half_width_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate raw-outcome interval half-widths at 50% coverage.

    Final infection proportion is converted from a proportion to percentage
    points so that the axis unit agrees with the SI text.
    """
    subset = data[
        np.isclose(data["Coverage"], TARGET_COVERAGE, atol=0.001)
        & data["Strategy"].isin(STRATEGY_ORDER)
    ].copy()

    frames = []

    for outcome_label, outcome_column in OUTCOMES.items():
        half_width = 1.96 * subset[f"{outcome_column} SE"]

        if outcome_label == "Final infection proportion":
            half_width = 100.0 * half_width

        frame = subset[
            ["VES_0", "R0", "Strategy"]
        ].copy()
        frame["Outcome"] = outcome_label
        frame["Interval half-width"] = half_width.to_numpy(dtype=float)
        frames.append(frame)

    output = pd.concat(frames, ignore_index=True)

    expected_rows = (
        85
        * len(STRATEGY_ORDER)
        * len(OUTCOMES)
    )
    if len(output) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} raw half-width records but "
            f"found {len(output)}."
        )

    return output


def prepare_strategy_difference_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate additional reductions and independent-mean intervals across the
    85 R0 x VE_S,0 settings at 50% coverage.
    """
    outcome_columns = list(OUTCOMES.values())

    baseline = data[
        (data["Strategy"] == "Random")
        & np.isclose(data["Coverage"], 0.0, atol=0.001)
    ][
        ["VES_0", "R0", *outcome_columns]
    ].copy()

    baseline = baseline.rename(
        columns={
            column: f"{column}__Y0"
            for column in outcome_columns
        }
    )

    random_columns = [
        "VES_0",
        "R0",
        *outcome_columns,
        *[f"{column} SE" for column in outcome_columns],
    ]

    random_target = data[
        (data["Strategy"] == "Random")
        & np.isclose(
            data["Coverage"],
            TARGET_COVERAGE,
            atol=0.001,
        )
    ][random_columns].copy()

    random_target = random_target.rename(
        columns={
            **{
                column: f"{column}__Random"
                for column in outcome_columns
            },
            **{
                f"{column} SE": f"{column} SE__Random"
                for column in outcome_columns
            },
        }
    )

    rows = []

    for strategy_label, strategy_name in TARGETED_STRATEGIES.items():
        targeted = data[
            (data["Strategy"] == strategy_name)
            & np.isclose(
                data["Coverage"],
                TARGET_COVERAGE,
                atol=0.001,
            )
        ][random_columns].copy()

        merged = (
            baseline
            .merge(
                random_target,
                on=["VES_0", "R0"],
                how="inner",
                validate="one_to_one",
            )
            .merge(
                targeted,
                on=["VES_0", "R0"],
                how="inner",
                validate="one_to_one",
            )
        )

        if len(merged) != 85:
            raise ValueError(
                f"Expected 85 parameter combinations for {strategy_label}, "
                f"but found {len(merged)}."
            )

        for outcome_label, outcome_column in OUTCOMES.items():
            y0 = merged[f"{outcome_column}__Y0"].to_numpy(dtype=float)
            y_random = merged[
                f"{outcome_column}__Random"
            ].to_numpy(dtype=float)
            y_strategy = merged[
                outcome_column
            ].to_numpy(dtype=float)

            se_random = merged[
                f"{outcome_column} SE__Random"
            ].to_numpy(dtype=float)
            se_strategy = merged[
                f"{outcome_column} SE"
            ].to_numpy(dtype=float)

            estimate = 100.0 * (
                y_random - y_strategy
            ) / y0

            propagated_se = 100.0 * np.sqrt(
                se_random ** 2 + se_strategy ** 2
            ) / y0

            lower = estimate - 1.96 * propagated_se
            upper = estimate + 1.96 * propagated_se

            direction = np.where(
                lower > 0,
                "Entirely above zero",
                np.where(
                    upper < 0,
                    "Entirely below zero",
                    "Includes zero",
                ),
            )

            frame = merged[["VES_0", "R0"]].copy()
            frame["Strategy"] = strategy_label
            frame["Outcome"] = outcome_label
            frame["Additional reduction (%)"] = estimate
            frame["Propagated SE"] = propagated_se
            frame["CI95 Lower"] = lower
            frame["CI95 Upper"] = upper
            frame["Direction"] = direction
            rows.append(frame)

    return pd.concat(rows, ignore_index=True)


def summarize_direction_counts(
    difference_data: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        difference_data
        .groupby(
            ["Strategy", "Outcome", "Direction"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "Count"})
    )

    complete_index = pd.MultiIndex.from_product(
        [
            list(TARGETED_STRATEGIES.keys()),
            list(OUTCOMES.keys()),
            DIRECTION_ORDER,
        ],
        names=["Strategy", "Outcome", "Direction"],
    )

    summary = (
        summary
        .set_index(["Strategy", "Outcome", "Direction"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    return summary


def validate_replicates(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Scenario Label",
        "Data Role",
        "VES_0",
        "R0",
        "Coverage",
        "Strategy",
        "MC_Run",
        "Infection Rate",
        "Peak Size",
        "Expected Deaths",
    }
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(
            "Replicate-level data are missing required columns: "
            + ", ".join(sorted(missing))
        )

    work = data.copy()
    for column in [
        "VES_0",
        "R0",
        "Coverage",
        "MC_Run",
        "Infection Rate",
        "Peak Size",
        "Expected Deaths",
    ]:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    return work


def mean_and_se(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    if len(array) < 2:
        raise ValueError(
            "At least two replicate outcomes are needed."
        )

    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1) / np.sqrt(len(array))),
    )


def make_forest_data(replicates: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate the five representative estimates from the retained
    replicate-level outcomes.
    """
    rows = []

    for case in REPRESENTATIVE_CASES:
        scenario = replicates[
            replicates["Scenario Label"]
            == case["Scenario Label"]
        ].copy()

        if scenario.empty:
            raise ValueError(
                "Representative replicate data were not found for "
                f"{case['Scenario Label']}."
            )

        outcome = case["Outcome"]
        strategy = case["Strategy"]

        y0_values = scenario[
            np.isclose(
                scenario["Coverage"],
                0.0,
                atol=0.001,
            )
            & (scenario["Strategy"] == "Random")
        ][outcome]

        random_values = scenario[
            np.isclose(
                scenario["Coverage"],
                TARGET_COVERAGE,
                atol=0.001,
            )
            & (scenario["Strategy"] == "Random")
        ][outcome]

        strategy_values = scenario[
            np.isclose(
                scenario["Coverage"],
                TARGET_COVERAGE,
                atol=0.001,
            )
            & (scenario["Strategy"] == strategy)
        ][outcome]

        if not (
            len(y0_values)
            == len(random_values)
            == len(strategy_values)
            == 100
        ):
            raise ValueError(
                f"Expected 100 runs for each component of "
                f"{case['Scenario Label']}, but obtained "
                f"{len(y0_values)}, {len(random_values)}, and "
                f"{len(strategy_values)}."
            )

        y0_mean, _ = mean_and_se(y0_values)
        random_mean, random_se = mean_and_se(random_values)
        strategy_mean, strategy_se = mean_and_se(
            strategy_values
        )

        estimate = 100.0 * (
            random_mean - strategy_mean
        ) / y0_mean

        propagated_se = 100.0 * np.sqrt(
            random_se ** 2 + strategy_se ** 2
        ) / y0_mean

        lower = estimate - 1.96 * propagated_se
        upper = estimate + 1.96 * propagated_se

        rows.append(
            {
                "Scenario Label": case["Scenario Label"],
                "Display Label": case["Display Label"],
                "Strategy": strategy,
                "Outcome": outcome,
                "Additional reduction (%)": estimate,
                "CI95 Lower": lower,
                "CI95 Upper": upper,
                "Y0 Mean": y0_mean,
                "Random Mean": random_mean,
                "Targeted Mean": strategy_mean,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 5. Plot helpers
# ============================================================

def deterministic_jitter(
    number_of_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(
        -0.14,
        0.14,
        size=number_of_points,
    )
    jitter -= jitter.mean()
    return jitter


def clean_axis(ax: plt.Axes) -> None:
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
        linewidth=0.5,
        alpha=0.22,
        zorder=0,
    )


def add_panel_letter(
    ax: plt.Axes,
    letter: str,
) -> None:
    if not SHOW_PANEL_LETTERS:
        return

    ax.text(
        -0.12,
        1.04,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


# ============================================================
# 6. Plot panels a-c
# ============================================================

def plot_half_width_panel(
    ax: plt.Axes,
    half_width_data: pd.DataFrame,
    outcome: str,
    panel_letter: str,
) -> None:
    panel = half_width_data[
        half_width_data["Outcome"] == outcome
    ].copy()

    positions = np.arange(len(STRATEGY_ORDER))
    values_by_strategy = [
        panel.loc[
            panel["Strategy"] == strategy,
            "Interval half-width",
        ].to_numpy(dtype=float)
        for strategy in STRATEGY_ORDER
    ]

    ax.boxplot(
        values_by_strategy,
        positions=positions,
        widths=0.50,
        patch_artist=False,
        showfliers=False,
        whis=1.5,
        medianprops={
            "linewidth": 1.2,
        },
        boxprops={
            "linewidth": 1.0,
        },
        whiskerprops={
            "linewidth": 0.9,
        },
        capprops={
            "linewidth": 0.9,
        },
    )

    for strategy_index, values in enumerate(
        values_by_strategy
    ):
        ax.scatter(
            np.full(len(values), positions[strategy_index])
            + deterministic_jitter(
                len(values),
                seed=20260726 + strategy_index,
            ),
            values,
            s=12,
            alpha=0.48,
            linewidth=0,
            zorder=2,
        )

        ax.scatter(
            positions[strategy_index],
            float(np.mean(values)),
            marker="D",
            s=34,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            STRATEGY_LABELS[strategy]
            for strategy in STRATEGY_ORDER
        ],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel(HALF_WIDTH_AXIS_LABELS[outcome])
    ax.set_title(outcome, pad=9)
    ax.set_xlim(-0.55, len(STRATEGY_ORDER) - 0.45)
    ax.set_ylim(bottom=0)

    clean_axis(ax)
    add_panel_letter(ax, panel_letter)


# ============================================================
# 7. Plot panel d (split by strategy; vertical orientation)
# ============================================================

def plot_direction_panel_for_strategy(
    ax: plt.Axes,
    direction_summary: pd.DataFrame,
    strategy_label: str,
    panel_letter: str,
) -> None:
    """
    Plot one vertical stacked-bar chart for a single strategy.

    The x-axis shows the three outcomes.
    The y-axis shows the number of parameter combinations (out of 85).
    """
    outcome_order = list(OUTCOMES.keys())
    x_positions = np.arange(len(outcome_order), dtype=float)

    bottom = np.zeros(len(outcome_order), dtype=float)

    for direction in DIRECTION_ORDER:
        counts = []
        for outcome in outcome_order:
            subset = direction_summary[
                (direction_summary["Strategy"] == strategy_label)
                & (direction_summary["Outcome"] == outcome)
                & (direction_summary["Direction"] == direction)
            ]
            if len(subset) != 1:
                raise ValueError(
                    f"Expected one row for strategy={strategy_label}, "
                    f"outcome={outcome}, direction={direction}, "
                    f"but found {len(subset)}."
                )
            counts.append(int(subset.iloc[0]["Count"]))

        counts = np.asarray(counts, dtype=float)

        bars = ax.bar(
            x_positions,
            counts,
            bottom=bottom,
            width=0.68,
            label=direction,
        )

        for bar, count, bar_bottom in zip(bars, counts, bottom):
            if count >= 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar_bottom + count / 2,
                    f"{int(count)}",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                )
            elif count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar_bottom + count + 1.2,
                    f"{int(count)}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

        bottom += counts

    strategy_title = (
        "High-exposure strategy"
        if strategy_label == "High-exposure strategy"
        else "Oldest-first strategy"
    )

    short_xlabels = [
        "Final infection\nproportion",
        "Epidemic peak\nsize",
        "Expected\ndeaths",
    ]

    ax.set_xticks(x_positions)
    ax.set_xticklabels(short_xlabels)
    ax.set_ylim(0, 90)
    ax.set_ylabel("Number of parameter combinations (of 85)")
    ax.set_title(
        f"Direction of approximate strategy-difference intervals\n{strategy_title}",
        pad=9,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.18),
        ncol=3,
        frameon=False,
        fontsize=8.2,
    )

    clean_axis(ax)
    add_panel_letter(ax, panel_letter)


# ============================================================
# 8. Plot panel e
# ============================================================

def plot_forest_panel(
    ax: plt.Axes,
    forest_data: pd.DataFrame,
    panel_letter: str,
) -> None:
    """
    Plot representative strategy comparisons as a vertical forest plot.

    The x-axis contains the five representative parameter settings.
    The y-axis contains additional reduction relative to the random strategy.
    Error bars show approximate 95% Monte Carlo intervals.
    """
    # Preserve the order defined in REPRESENTATIVE_CASES so that the five
    # representative settings appear from left to right in the intended order.
    plot_data = forest_data.reset_index(drop=True)
    x_positions = np.arange(len(plot_data), dtype=float)

    lower_bound = min(
        float(plot_data["CI95 Lower"].min()),
        0.0,
    )
    upper_bound = max(
        float(plot_data["CI95 Upper"].max()),
        0.0,
    )
    span = upper_bound - lower_bound
    if span <= 0:
        span = 1.0

    label_offset = 0.035 * span

    for row_index, row in plot_data.iterrows():
        estimate = float(row["Additional reduction (%)"])
        lower = float(row["CI95 Lower"])
        upper = float(row["CI95 Upper"])

        marker = (
            "o"
            if row["Strategy"] == "High Strength (Node)"
            else "s"
        )

        ax.errorbar(
            x_positions[row_index],
            estimate,
            yerr=np.array(
                [
                    [estimate - lower],
                    [upper - estimate],
                ]
            ),
            fmt=marker,
            markersize=6,
            capsize=3,
            linewidth=1.2,
        )

        # Place interval text above positive estimates and below negative
        # estimates so that labels do not obscure the point or interval.
        if estimate >= 0:
            text_y = upper + label_offset
            vertical_alignment = "bottom"
        else:
            text_y = lower - label_offset
            vertical_alignment = "top"

        ax.text(
            x_positions[row_index],
            text_y,
            f"{estimate:.2f}\n[{lower:.2f}, {upper:.2f}]",
            ha="center",
            va=vertical_alignment,
            fontsize=7.6,
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=0.9,
        alpha=0.65,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        plot_data["Display Label"],
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel(
        "Additional reduction relative to random strategy (%)"
    )
    ax.set_xlabel(
        "Representative parameter setting"
    )
    ax.set_title(
        "Representative strategy comparisons",
        pad=9,
    )

    # Add sufficient vertical space for the numerical interval labels.
    ax.set_ylim(
        lower_bound - 0.15 * span,
        upper_bound + 0.18 * span,
    )
    ax.set_xlim(
        -0.55,
        len(plot_data) - 0.45,
    )

    clean_axis(ax)
    add_panel_letter(ax, panel_letter)


# ============================================================
# 9. Separate panel output
# ============================================================

def save_panel(
    fig: plt.Figure,
    panel_key: str,
) -> None:
    """Save one panel as an independent PDF and PNG."""
    fig.savefig(
        PANEL_OUTPUTS[panel_key]["pdf"],
        bbox_inches="tight",
    )
    fig.savefig(
        PANEL_OUTPUTS[panel_key]["png"],
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_separate_panels(
    half_width_data: pd.DataFrame,
    direction_summary: pd.DataFrame,
    forest_data: pd.DataFrame,
) -> None:
    """
    Export panels separately for manual assembly in PowerPoint.

    Panels:
        a  Final infection proportion uncertainty
        b  Epidemic peak size uncertainty
        c  Expected deaths uncertainty
        d  Direction classification for high-exposure strategy
        e  Direction classification for oldest-first strategy
        S8 Representative forest plot

    No combined multi-panel figure is produced.
    """
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig_a, ax_a = plt.subplots(figsize=(5.2, 4.3))
    plot_half_width_panel(ax_a, half_width_data, "Final infection proportion", "a")
    fig_a.subplots_adjust(left=0.20, right=0.97, bottom=0.25, top=0.90)
    save_panel(fig_a, "a")

    fig_b, ax_b = plt.subplots(figsize=(5.2, 4.3))
    plot_half_width_panel(ax_b, half_width_data, "Epidemic peak size", "b")
    fig_b.subplots_adjust(left=0.20, right=0.97, bottom=0.25, top=0.90)
    save_panel(fig_b, "b")

    fig_c, ax_c = plt.subplots(figsize=(5.2, 4.3))
    plot_half_width_panel(ax_c, half_width_data, "Expected deaths", "c")
    fig_c.subplots_adjust(left=0.20, right=0.97, bottom=0.25, top=0.90)
    save_panel(fig_c, "c")

    fig_d1, ax_d1 = plt.subplots(figsize=(7.0, 5.0))
    plot_direction_panel_for_strategy(
        ax_d1, direction_summary, "High-exposure strategy", "d"
    )
    fig_d1.subplots_adjust(left=0.13, right=0.98, bottom=0.28, top=0.88)
    save_panel(fig_d1, "d")

    fig_d2, ax_d2 = plt.subplots(figsize=(7.0, 5.0))
    plot_direction_panel_for_strategy(
        ax_d2, direction_summary, "Oldest-first strategy", "e"
    )
    fig_d2.subplots_adjust(left=0.13, right=0.98, bottom=0.28, top=0.88)
    save_panel(fig_d2, "e")

    fig_e, ax_e = plt.subplots(figsize=(10.2, 6.4))
    plot_forest_panel(ax_e, forest_data, "")
    fig_e.subplots_adjust(left=0.13, right=0.98, bottom=0.34, top=0.90)
    save_panel(fig_e, "s8")


# ============================================================
# 10. Main
# ============================================================

def main() -> None:
    main_summary_raw, main_source = load_main_summary()
    replicate_raw, replicate_source = load_selected_replicates()

    main_summary = validate_main_summary(main_summary_raw)
    replicates = validate_replicates(replicate_raw)

    half_width_data = make_half_width_data(main_summary)
    strategy_difference_data = (
        prepare_strategy_difference_data(main_summary)
    )
    direction_summary = summarize_direction_counts(
        strategy_difference_data
    )
    forest_data = make_forest_data(replicates)

    half_width_data.to_csv(
        OUTPUT_HALF_WIDTH_DATA,
        index=False,
    )
    direction_summary.to_csv(
        OUTPUT_DIRECTION_DATA,
        index=False,
    )
    forest_data.to_csv(
        OUTPUT_FOREST_DATA,
        index=False,
    )

    plot_separate_panels(
        half_width_data,
        direction_summary,
        forest_data,
    )

    print(f"[INFO] Main summary: {main_source}")
    print(f"[INFO] Selected replicates: {replicate_source}")
    print("[INFO] Direction classification:")
    print(
        direction_summary.to_string(index=False)
    )
    print("[INFO] Representative intervals:")
    print(
        forest_data[
            [
                "Scenario Label",
                "Additional reduction (%)",
                "CI95 Lower",
                "CI95 Upper",
            ]
        ].to_string(index=False)
    )
    for panel_key in ["a", "b", "c", "d", "e", "s8"]:
        print(f"[SAVED] {PANEL_OUTPUTS[panel_key]['pdf']}")
        print(f"[SAVED] {PANEL_OUTPUTS[panel_key]['png']}")
    print(f"[SAVED] {OUTPUT_HALF_WIDTH_DATA}")
    print(f"[SAVED] {OUTPUT_DIRECTION_DATA}")
    print(f"[SAVED] {OUTPUT_FOREST_DATA}")


if __name__ == "__main__":
    main()
