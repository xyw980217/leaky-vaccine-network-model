"""Plot outcome differences between targeted and random vaccination strategies.

The script reads the primary simulation summary at 50% vaccination coverage
and generates one three-outcome heatmap set for each targeted strategy.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    PROJECT_ROOT / "results" / "primary_simulation"
    / "Simulation_Results_All_Scenarios_Parallel.xlsx"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "main_strategy_heatmaps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
file_path = INPUT_FILE

try:
    df = pd.read_excel(file_path)
    print(f"[INFO] Loaded data file: {file_path}")
except FileNotFoundError:
    print(f"\n[ERROR] File not found: '{file_path}'")
    sys.exit(1)

df['Coverage'] = df['Coverage'].round(2)

target_strategies = [
    'Random',
    'High Strength (Node)',
    'Priority: Oldest First',
    'Household Priority: Hub-based',
]
metrics = ['Infection Rate', 'Peak Size', 'Expected Deaths']

df_filtered = df[df['Coverage'].isin([0.0, 0.5])]
df_filtered = df_filtered[df_filtered['Strategy'].isin(target_strategies)]

df_0 = df_filtered[
    (df_filtered['Coverage'] == 0.0) &
    (df_filtered['Strategy'] == 'Random')
].set_index(['VES_0', 'R0'])[metrics]

df_50 = df_filtered[
    df_filtered['Coverage'] == 0.5
].set_index(['VES_0', 'R0', 'Strategy'])[metrics]

df_0.columns = [f"{m}_Baseline" for m in metrics]

drop_pct = pd.merge(
    df_50.reset_index(),
    df_0.reset_index(),
    on=['VES_0', 'R0']
)

for m in metrics:
    drop_pct[m] = (
        (drop_pct[f"{m}_Baseline"] - drop_pct[m])
        / drop_pct[f"{m}_Baseline"]
        * 100
    )

drop_pct = drop_pct[
    ['VES_0', 'R0', 'Strategy'] + metrics
].fillna(0)

random_drop = drop_pct[drop_pct['Strategy'] == 'Random'].set_index(['VES_0', 'R0'])[metrics]
random_drop.columns = [f"{m}_Random" for m in metrics]

merged = pd.merge(drop_pct[drop_pct['Strategy'] != 'Random'], random_drop, on=['VES_0', 'R0'])
for m in metrics:
    merged[f"{m}_Extra_Drop"] = merged[m] - merged[f"{m}_Random"]

global_bounds = {}
for m in metrics:
    extra_metric_name = f"{m}_Extra_Drop"
    abs_max_val = max(abs(merged[extra_metric_name].max()), abs(merged[extra_metric_name].min()))
    
    if abs_max_val < 0.1:
        abs_max_val = 0.5
        
    global_bounds[m] = (-abs_max_val, abs_max_val)
    print(f"Outcome: {m:20s}, shared colour range: [{global_bounds[m][0]:.2f}, {global_bounds[m][1]:.2f}]")
# ==============================================================================

strategy_cols = [s for s in target_strategies if s != 'Random']

sns.set_theme(style="white")

for i, strategy in enumerate(strategy_cols):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f"Strategy: {strategy} (Extra Reduction % vs Random)", fontsize=18, fontweight='bold', y=0.98)
    
    for j, m in enumerate(metrics):
        pivot_data = merged[merged['Strategy'] == strategy].pivot(index='VES_0', columns='R0', values=f"{m}_Extra_Drop")
        pivot_data = pivot_data.sort_index(ascending=False) 
        
        vmin_global, vmax_global = global_bounds[m]
        
        sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="RdYlGn", 
                    vmin=vmin_global, vmax=vmax_global, center=0, 
                    ax=axes[j], cbar_kws={'label': 'Extra Drop (%)'}, linewidths=.5)
        
        axes[j].set_title(f"{m}", fontsize=15, pad=10)
        axes[j].set_ylabel("Initial Vaccine Efficacy (VES_0)", fontsize=11)
        
        if j == 2:
            axes[j].set_xlabel("Basic Reproduction Number (R0)", fontsize=12)
        else:
            axes[j].set_xlabel("")
            
        axes[j].tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    
    filename_by_strategy = {
        "High Strength (Node)": "Figure3_heatmap_components.png",
        "Priority: Oldest First": "Figure5_heatmap_components.png",
        "Household Priority: Hub-based": (
            "Household_Priority_Hub_Based_heatmap_components.png"
        ),
    }
    filename = filename_by_strategy[strategy]
    png_path = OUTPUT_DIR / filename
    pdf_path = png_path.with_suffix(".pdf")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")