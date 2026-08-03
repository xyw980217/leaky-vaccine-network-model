"""Plot the exposure-bin decomposition used for main Figure 4.

The script reads exact bin-level outputs from the primary simulation, checks
consistency with the scenario summaries, and exports the two mechanism panels,
a shared legend, and the underlying summary table.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import PchipInterpolator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Paths and target scenario
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / 'results' / 'primary_simulation'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'figures' / 'exposure_distribution_mechanism'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
current_path = INPUT_ROOT

target_coverage = 0.5
target_strategy = "High Strength (Node)"
baseline_strategy = "Random"

ves0 = 0.8
r0 = 3.0
coverage_label = int(round(target_coverage * 100))

bin_file_basename = (
    f"Feature_Bin_Exact_AllCoverages_VES{ves0}_R0_{r0}.csv.gz"
)
scenario_file_basename = (
    f"Scenario_Summary_VES{ves0}_R0_{r0}.csv"
)
simulation_csv_basename = (
    "Simulation_Results_All_Scenarios_With_Uncertainty.csv.gz"
)
simulation_excel_basename = (
    "Simulation_Results_All_Scenarios_Parallel.xlsx"
)

search_roots = [
    current_path,
    current_path.parent,
    current_path.parent.parent
]


def _find_existing_file(candidates, description):
    """Return the first existing path and report all checked paths on failure."""
    unique_candidates = []
    seen = set()

    for candidate in candidates:
        candidate = Path(candidate).resolve()
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"  - {p}" for p in unique_candidates)
    raise FileNotFoundError(
        f"Could not find {description}. Searched the following locations:\n{searched}\n"
        "Confirm that the primary simulation completed and that VES_0 and R0 match the requested setting."
    )


bin_candidates = []
scenario_candidates = []
simulation_candidates = []

for root in search_roots:
    bin_candidates.extend([
        root / bin_file_basename,
        root / "fig4_exact_bin_counts" / bin_file_basename,
        root / "retained_simulation_data"
             / "fig4_exact_bin_counts"
             / bin_file_basename
    ])

    scenario_candidates.extend([
        root / scenario_file_basename,
        root / "scenario_summaries" / scenario_file_basename,
        root / "retained_simulation_data"
             / "scenario_summaries"
             / scenario_file_basename
    ])

    simulation_candidates.extend([
        root / simulation_csv_basename,
        root / simulation_excel_basename,
        root / "retained_simulation_data" / simulation_csv_basename,
        root / "retained_simulation_data" / simulation_excel_basename
    ])

file_name = _find_existing_file(
    bin_candidates,
    "the exact exposure-bin file for Fig. 4"
)
scenario_file_name = _find_existing_file(
    scenario_candidates,
    "the target-scenario summary file"
)
simulation_file_name = _find_existing_file(
    simulation_candidates,
    "the combined scenario-summary file"
)

show_title = False

main_output_a = OUTPUT_DIR / (
    f"Fig4a_Uninfected_Proportion_Difference_"
    f"VES{ves0:.1f}_R{r0:.1f}_Cov{coverage_label}.png"
)
main_output_b = OUTPUT_DIR / (
    f"Fig4b_Weighted_Bin_Contribution_"
    f"VES{ves0:.1f}_R{r0:.1f}_Cov{coverage_label}.png"
)
legend_output = OUTPUT_DIR / (
    f"Fig4_Standalone_Legend_"
    f"VES{ves0:.1f}_R{r0:.1f}_Cov{coverage_label}.png"
)
summary_output = OUTPUT_DIR / (
    f"Fig4_Bin_Summary_"
    f"VES{ves0:.1f}_R{r0:.1f}_Cov{coverage_label}.xlsx"
)
main_output_a_pdf = main_output_a.with_suffix(".pdf")
main_output_b_pdf = main_output_b.with_suffix(".pdf")
legend_output_pdf = legend_output.with_suffix(".pdf")

# ============================================
# ============================================
print("[INFO] Input files:")
print(f"  Exact bin data: {file_name}")
print(f"  Scenario summary: {scenario_file_name}")
print(f"  All-scenario summary: {simulation_file_name}")

df = pd.read_csv(file_name, compression="gzip")
scenario_df = pd.read_csv(scenario_file_name)

if str(simulation_file_name).lower().endswith(".csv.gz"):
    simulation_df = pd.read_csv(
        simulation_file_name,
        compression="gzip"
    )
else:
    simulation_df = pd.read_excel(simulation_file_name)

required_columns = {
    "VES_0",
    "R0",
    "Coverage",
    "Strategy",
    "MC_Run",
    "Bin_ID",
    "Bin_Type",
    "Bin_Left",
    "Bin_Right",
    "Bin_Center",
    "Total_Nodes",
    "Infected_Nodes",
    "Uninfected_Nodes",
    "Uninfected_Proportion",
    "Node_Weight"
}
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        "The exact exposure-bin file is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )

scenario_required_columns = {
    "VES_0",
    "R0",
    "Coverage",
    "Strategy",
    "Infection Rate"
}
scenario_missing_columns = scenario_required_columns.difference(
    scenario_df.columns
)

if scenario_missing_columns:
    raise ValueError(
        "Scenario_Summary 文件缺少必要列："
        + ", ".join(sorted(scenario_missing_columns))
    )

# ============================================
# ============================================
df_sub = df[
    np.isclose(
        pd.to_numeric(df["Coverage"], errors="coerce"),
        target_coverage,
        atol=1e-8
    )
    & np.isclose(
        pd.to_numeric(df["VES_0"], errors="coerce"),
        ves0,
        atol=1e-8
    )
    & np.isclose(
        pd.to_numeric(df["R0"], errors="coerce"),
        r0,
        atol=1e-8
    )
].copy()

df_sub = df_sub[
    df_sub["Strategy"].isin(
        [target_strategy, baseline_strategy]
    )
].copy()

if df_sub.empty:
    raise ValueError(
        "在 Feature_Bin_Exact 文件中没有找到目标场景数据。"
    )

available_strategies = set(df_sub["Strategy"].dropna().unique())
required_strategies = {target_strategy, baseline_strategy}
missing_strategies = required_strategies.difference(
    available_strategies
)

if missing_strategies:
    raise ValueError(
        "缺少策略："
        + ", ".join(sorted(missing_strategies))
    )

mc_counts = (
    df_sub.groupby("Strategy")["MC_Run"]
    .nunique()
)

if mc_counts.nunique() != 1:
    raise ValueError(
        "两种策略的 Monte Carlo 重复数不一致：\n"
        + mc_counts.to_string()
    )

print("[INFO] Monte Carlo repetitions per strategy:")
print(mc_counts.to_string())

# ============================================
# ============================================
summary = (
    df_sub.groupby(
        ["Strategy", "Bin_ID"],
        observed=False
    )
    .agg(
        bin_type=("Bin_Type", "first"),
        bin_left=("Bin_Left", "first"),
        bin_right=("Bin_Right", "first"),
        bin_center=("Bin_Center", "first"),
        total_nodes=("Total_Nodes", "first"),
        node_weight=("Node_Weight", "first"),
        mean_uninfected_proportion=(
            "Uninfected_Proportion",
            "mean"
        ),
        sd_uninfected_proportion=(
            "Uninfected_Proportion",
            "std"
        ),
        n_mc=("Uninfected_Proportion", "count")
    )
    .reset_index()
)

baseline_summary = summary[
    summary["Strategy"] == baseline_strategy
].set_index("Bin_ID")

target_summary = summary[
    summary["Strategy"] == target_strategy
].set_index("Bin_ID")

common_bin_ids = baseline_summary.index.intersection(
    target_summary.index
)

if len(common_bin_ids) == 0:
    raise ValueError("两种策略之间没有可比较的共同 bin。")

baseline_summary = baseline_summary.loc[common_bin_ids]
target_summary = target_summary.loc[common_bin_ids]

for col in [
    "bin_left",
    "bin_right",
    "bin_center",
    "total_nodes",
    "node_weight"
]:
    if not np.allclose(
        pd.to_numeric(
            baseline_summary[col],
            errors="coerce"
        ),
        pd.to_numeric(
            target_summary[col],
            errors="coerce"
        ),
        equal_nan=True
    ):
        raise ValueError(
            f"两种策略的 {col} 不一致，不能直接比较。"
        )

plot_df = pd.DataFrame({
    "bin_id": common_bin_ids,
    "bin_type": target_summary["bin_type"].values,
    "bin_left": target_summary["bin_left"].values,
    "bin_right": target_summary["bin_right"].values,
    "bin_center": target_summary["bin_center"].values,
    "total_nodes": target_summary["total_nodes"].values,
    "node_weight": target_summary["node_weight"].values,
    "baseline_mean": (
        baseline_summary[
            "mean_uninfected_proportion"
        ].values
    ),
    "target_mean": (
        target_summary[
            "mean_uninfected_proportion"
        ].values
    ),
    "baseline_sd": (
        baseline_summary[
            "sd_uninfected_proportion"
        ].values
    ),
    "target_sd": (
        target_summary[
            "sd_uninfected_proportion"
        ].values
    ),
    "baseline_n": baseline_summary["n_mc"].values,
    "target_n": target_summary["n_mc"].values
}).sort_values("bin_id")

plot_df["rate_diff"] = (
    plot_df["target_mean"]
    - plot_df["baseline_mean"]
)

plot_df["rate_diff_se"] = np.sqrt(
    (plot_df["target_sd"] ** 2)
    / plot_df["target_n"]
    +
    (plot_df["baseline_sd"] ** 2)
    / plot_df["baseline_n"]
)

plot_df["rate_diff_ci95"] = (
    1.96 * plot_df["rate_diff_se"]
)

plot_df["weighted_contribution"] = (
    plot_df["node_weight"]
    * plot_df["rate_diff"]
)

plot_df["weighted_contribution_se"] = (
    plot_df["node_weight"]
    * plot_df["rate_diff_se"]
)

plot_df["weighted_contribution_ci95"] = (
    1.96
    * plot_df["weighted_contribution_se"]
)

plot_df["rate_diff_pp"] = (
    plot_df["rate_diff"] * 100
)
plot_df["rate_diff_ci95_pp"] = (
    plot_df["rate_diff_ci95"] * 100
)
plot_df["weighted_contribution_pp"] = (
    plot_df["weighted_contribution"] * 100
)
plot_df["weighted_contribution_ci95_pp"] = (
    plot_df["weighted_contribution_ci95"] * 100
)

# ============================================
# ============================================
scenario_sub = scenario_df[
    np.isclose(
        pd.to_numeric(
            scenario_df["Coverage"],
            errors="coerce"
        ),
        target_coverage,
        atol=1e-8
    )
    & np.isclose(
        pd.to_numeric(
            scenario_df["VES_0"],
            errors="coerce"
        ),
        ves0,
        atol=1e-8
    )
    & np.isclose(
        pd.to_numeric(
            scenario_df["R0"],
            errors="coerce"
        ),
        r0,
        atol=1e-8
    )
    & scenario_df["Strategy"].isin(
        [target_strategy, baseline_strategy]
    )
].copy()

if len(scenario_sub) != 2:
    raise ValueError(
        "Scenario_Summary 中Could not find 恰好两条目标策略记录。"
    )

scenario_rates = (
    scenario_sub.set_index("Strategy")[
        "Infection Rate"
    ]
)

overall_uninfected_diff = (
    (
        1.0
        - scenario_rates.loc[target_strategy]
    )
    -
    (
        1.0
        - scenario_rates.loc[baseline_strategy]
    )
)

bin_contribution_sum = (
    plot_df["weighted_contribution"].sum()
)

if not np.isclose(
    bin_contribution_sum,
    overall_uninfected_diff,
    atol=1e-10,
    rtol=1e-8
):
    raise RuntimeError(
        "bin 加权贡献之和与总体未感染比例差异不一致：\n"
        f"Σ bin contributions = "
        f"{bin_contribution_sum * 100:.10f} pp\n"
        f"Overall difference = "
        f"{overall_uninfected_diff * 100:.10f} pp"
    )

simulation_sub = simulation_df[
    np.isclose(
        pd.to_numeric(
            simulation_df["Coverage"],
            errors="coerce"
        ),
        target_coverage,
        atol=1e-8
    )
    & np.isclose(
        pd.to_numeric(
            simulation_df["VES_0"],
            errors="coerce"
        ),
        ves0,
        atol=1e-8
    )
    & np.isclose(
        pd.to_numeric(
            simulation_df["R0"],
            errors="coerce"
        ),
        r0,
        atol=1e-8
    )
    & simulation_df["Strategy"].isin(
        [target_strategy, baseline_strategy]
    )
].copy()

if len(simulation_sub) == 2:
    scenario_check = (
        scenario_sub
        .sort_values("Strategy")[
            "Infection Rate"
        ]
        .to_numpy()
    )
    simulation_check = (
        simulation_sub
        .sort_values("Strategy")[
            "Infection Rate"
        ]
        .to_numpy()
    )

    if not np.allclose(
        scenario_check,
        simulation_check,
        atol=1e-12,
        rtol=1e-10
    ):
        raise RuntimeError(
            "the combined scenario-summary file与 "
            "Scenario_Summary 的 Infection Rate 不一致。"
        )

print("\n[CHECK] Overall consistency:")
print(
    "  Sum of all bin contributions = "
    f"{bin_contribution_sum * 100:.6f} pp"
)
print(
    "  Overall uninfected-proportion difference = "
    f"{overall_uninfected_diff * 100:.6f} pp"
)
print(
    "  Overall final-infection-proportion difference "
    "(High Strength - Random) = "
    f"{-overall_uninfected_diff * 100:.6f} pp"
)

plot_df.to_excel(summary_output, index=False)

# ============================================
# ============================================
positive_plot_df = plot_df[
    (plot_df["bin_id"] > 0)
    & (plot_df["bin_center"] > 0)
].copy()

if len(positive_plot_df) < 2:
    raise ValueError(
        "正 strength 的有效 bin 少于2个，无法绘图。"
    )

zero_bin_df = plot_df[
    plot_df["bin_id"] == 0
].copy()

if not zero_bin_df.empty:
    print(
        "\n[INFO] Bin_ID=0 is excluded from the log-scale "
        "x-axis but retained in the overall validation:"
    )
    print(
        f"  Nodes = "
        f"{int(zero_bin_df['total_nodes'].iloc[0])}"
    )
    print(
        "  Contribution = "
        f"{zero_bin_df['weighted_contribution_pp'].iloc[0]:.6f} pp"
    )

# ============================================
# ============================================
sns.set_theme(style="whitegrid", context="paper")

x = positive_plot_df["bin_center"].to_numpy()
y = positive_plot_df["rate_diff_pp"].to_numpy()
yerr = positive_plot_df[
    "rate_diff_ci95_pp"
].to_numpy()

x_log = np.log10(x)
interp = PchipInterpolator(x_log, y)
x_log_dense = np.linspace(
    x_log.min(),
    x_log.max(),
    800
)
x_dense = 10 ** x_log_dense
y_dense = interp(x_log_dense)

fig_a, ax_a = plt.subplots(figsize=(8.5, 6))

ax_a.axhline(
    0,
    color="black",
    linewidth=1.2
)

ax_a.plot(
    x_dense,
    y_dense,
    color="black",
    linewidth=2.0
)

ax_a.fill_between(
    x_dense,
    0,
    y_dense,
    where=(y_dense > 0),
    interpolate=True,
    color="#55a868",
    alpha=0.45
)

ax_a.fill_between(
    x_dense,
    0,
    y_dense,
    where=(y_dense < 0),
    interpolate=True,
    color="#4c72b0",
    alpha=0.45
)

ax_a.errorbar(
    x,
    y,
    yerr=yerr,
    fmt="o",
    color="black",
    ecolor="black",
    elinewidth=1.0,
    capsize=3,
    markersize=4.5,
    zorder=4
)

ax_a.set_xscale("log")
ax_a.set_xlabel(
    "Total weighted exposure strength",
    fontsize=14,
    fontweight="bold",
    labelpad=10
)
ax_a.set_ylabel(
    "Uninfected-proportion difference (pp)",
    fontsize=14,
    fontweight="bold",
    labelpad=10
)

if show_title:
    ax_a.set_title(
        "Difference in uninfected proportion across "
        "exposure-strength bins\n"
        f"({target_strategy} - {baseline_strategy}; "
        f"coverage={coverage_label}%, "
        f"$R_0$={r0:.1f}, "
        f"$VE_{{S,0}}$={ves0:.1f})",
        fontsize=15,
        fontweight="bold",
        pad=14
    )

plt.setp(
    ax_a.get_xticklabels(),
    fontsize=12
)
plt.setp(
    ax_a.get_yticklabels(),
    fontsize=12
)

ax_a.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.35
)

fig_a.tight_layout()
fig_a.savefig(
    main_output_a,
    dpi=300,
    bbox_inches="tight"
)
fig_a.savefig(
    main_output_a_pdf,
    bbox_inches="tight"
)
plt.show()
plt.close(fig_a)

# ============================================
# ============================================
fig_b, ax_b = plt.subplots(figsize=(8.5, 6))

bar_colors = np.where(
    positive_plot_df[
        "weighted_contribution_pp"
    ] >= 0,
    "#55a868",
    "#4c72b0"
)

ax_b.bar(
    positive_plot_df["bin_left"],
    positive_plot_df[
        "weighted_contribution_pp"
    ],
    width=(
        positive_plot_df["bin_right"]
        - positive_plot_df["bin_left"]
    ),
    align="edge",
    color=bar_colors,
    alpha=0.60,
    edgecolor="black",
    linewidth=0.8
)

ax_b.errorbar(
    positive_plot_df["bin_center"],
    positive_plot_df[
        "weighted_contribution_pp"
    ],
    yerr=positive_plot_df[
        "weighted_contribution_ci95_pp"
    ],
    fmt="none",
    ecolor="black",
    elinewidth=1.0,
    capsize=3,
    zorder=4
)

ax_b.axhline(
    0,
    color="black",
    linewidth=1.2
)

ax_b.set_xscale("log")
ax_b.set_xlabel(
    "Total weighted exposure strength",
    fontsize=14,
    fontweight="bold",
    labelpad=10
)
ax_b.set_ylabel(
    "Contribution to overall difference (pp)",
    fontsize=14,
    fontweight="bold",
    labelpad=10
)

if show_title:
    ax_b.set_title(
        "Population-weighted contribution of each "
        "exposure-strength bin\n"
        f"({target_strategy} - {baseline_strategy}; "
        f"coverage={coverage_label}%, "
        f"$R_0$={r0:.1f}, "
        f"$VE_{{S,0}}$={ves0:.1f})",
        fontsize=15,
        fontweight="bold",
        pad=14
    )

plt.setp(
    ax_b.get_xticklabels(),
    fontsize=12
)
plt.setp(
    ax_b.get_yticklabels(),
    fontsize=12
)

ax_b.grid(
    True,
    axis="y",
    linestyle="--",
    alpha=0.35
)
ax_b.grid(
    False,
    axis="x"
)

fig_b.tight_layout()
fig_b.savefig(
    main_output_b,
    dpi=300,
    bbox_inches="tight"
)
fig_b.savefig(
    main_output_b_pdf,
    bbox_inches="tight"
)
plt.show()
plt.close(fig_b)

# ============================================
# ============================================
fig_leg = plt.figure(figsize=(11, 0.8))

legend_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        marker="o",
        linestyle="none",
        markersize=5,
        markerfacecolor="black",
        markeredgecolor="black",
        label="Mean estimate with 95% CI"
    ),
    Patch(
        facecolor="#55a868",
        edgecolor="black",
        alpha=0.60,
        label="Favours high-exposure strategy"
    ),
    Patch(
        facecolor="#4c72b0",
        edgecolor="black",
        alpha=0.60,
        label="Favours random strategy"
    )
]

fig_leg.legend(
    handles=legend_handles,
    loc="center",
    ncol=3,
    frameon=False,
    fontsize=11
)

plt.axis("off")
fig_leg.tight_layout()
fig_leg.savefig(
    legend_output,
    dpi=300,
    bbox_inches="tight"
)
fig_leg.savefig(
    legend_output_pdf,
    bbox_inches="tight"
)
plt.show()
plt.close(fig_leg)

print("\nPlotting completed:")
print(f"  Fig. 4a: {main_output_a}")
print(f"  Fig. 4b: {main_output_b}")
print(f"  Legend: {legend_output}")
print(f"  Summary data: {summary_output}")
