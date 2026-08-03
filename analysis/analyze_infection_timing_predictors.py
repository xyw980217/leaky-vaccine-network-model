"""Analyse simulated infection timing and its node-level predictors.

The workflow constructs the survey-derived exposure network, runs the
unvaccinated reference simulations, and performs distributional, linear,
random-forest, and SHAP analyses of mean first infection time.
"""

import os
from pathlib import Path
import re
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import get_close_matches
from collections import defaultdict
from scipy import stats
from scipy.sparse.linalg import eigs

# ML Libraries
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP library not installed. SHAP analysis will be skipped.")

# =============== Parameters ===============
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = '20231012.xlsx'
DATA_PATH = str(PROJECT_ROOT / 'data' / DATA_FILE)
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'infection_timing_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHEET_ID = 0

# ---- SEIS Parameters ----
TARGET_R0 = 7.0
BETA_EDGE = None 
PROB_E_TO_I = 0.2 #0.142
PROB_I_TO_S = 0.142

# Piecewise natural-immunity efficacy
NAT_EFF_MAX = 0.652
NAT_EFF_MIN = 0.247

# ---- Network & Time Parameters ----
HH_CONTACT_HOURS = 12.0 
SOCIAL_ENCOUNTER_DURATION = 0.25
INTERNAL_SOCIAL_RATIO = 0.3

# ---- External Virtual Parameters ----
VIRTUAL_CONTACT_MAPPING = {'a': 3.0, 'b': 10.0, 'c': 20.0, 'd': 30.0}
EXTERNAL_PREVALENCE = 0.005 
EXT_SOCIAL_CONTACT_DENSITY = 4.0 

# ---- Monte Carlo ----
MC_RUNS_TOTAL = 500
SIMULATION_DAYS = 500
NUM_SEEDS_PER_RUN = 10
RNG_SEED = 42

np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

# =============== Helper Functions ===============
def _normalize_col(s):
    s = str(s).replace('\ufeff', '').strip().replace(' ', '')
    return s.replace('（', '(').replace('）', ')').replace('/', '')

def pick_column(df, targets):
    norm_map = {col: _normalize_col(col) for col in df.columns}
    wanted_norm = [_normalize_col(t) for t in targets]
    for col, ncol in norm_map.items():
        if ncol in wanted_norm: return col
    for col, ncol in norm_map.items():
        if any(w in ncol for w in wanted_norm) or any(ncol in w for w in wanted_norm): return col
    best = None
    for w in wanted_norm:
        cands = get_close_matches(w, list(norm_map.values()), n=1, cutoff=0.6)
        if cands:
            for k, v in norm_map.items():
                if v == cands[0]: return k
    raise KeyError(f"Column not found: {targets}")

FREQ_MAP = {'无': 0, '<1次': 0.5, '1-2次': 1.5, '3-4次': 3.5, '≥5次': 5.5, 'a': 5.5, 'b': 3.5, 'c': 1.5, 'd': 0.5}
DUR_MAP = {'无': 0, '2小时以内': 1, '2-5小时': 3.5, '5小时以上': 6, 'a': 1, 'b': 3.5, 'c': 6}

def _map_freq(x): s=str(x).strip(); return FREQ_MAP.get(s) if s in FREQ_MAP else (5.5 if '5次' in s else 0.5)
def _map_dur(x): s=str(x).strip(); return DUR_MAP.get(s) if s in DUR_MAP else (3.5 if '2-5' in s else 0.0)
def _infer_employed(s): s=str(s).strip(); return s in ['b', 'B', 'd', 'D'] or '工' in s or '班' in s
def _infer_student(s): s=str(s).strip(); return '学' in s or s in ['a', 'A']
def _map_contact_count(x):
    s = str(x).lower().strip()
    for k, v in VIRTUAL_CONTACT_MAPPING.items():
        if k in s: return v
    return 0.0 
# =============== Data Loading ===============
if not os.path.exists(DATA_PATH): raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_ID)
N = len(df)
print(f"[INFO] Loaded N = {N}")

COL_COMMUNITY = pick_column(df, ["小区名称", "小区", "社区名称", "社区"])
COL_ADDRESS = pick_column(df, ["住户详细地址", "住址"])
COL_HH_INDEX = pick_column(df, ["本户第份"])
COL_WORK_STATUS = pick_column(df, ["工作状态"])
COL_CONTACT_COUNT = pick_column(df, ["人员数量", "学习场所的人员数量"])
COL_TRIP_FREQ = [f"频率{i}" for i in range(1, 6)]
COL_TRIP_DUR = [f"时长{i}" for i in range(1, 6)]
COL_AGE = pick_column(df, ["年龄"])

# ---------------- Household identifier ----------------
# Household definition:
def _clean_household_field(series):
    return (
        series.fillna('')
              .astype(str)
              .str.replace('\ufeff', '', regex=False)
              .str.replace('／', '/', regex=False)
              .str.replace(r'\s+', '', regex=True)
              .str.strip()
    )

df['_community_clean'] = _clean_household_field(df[COL_COMMUNITY])
df['_address_clean'] = _clean_household_field(df[COL_ADDRESS])
df['_hh_index_clean'] = _clean_household_field(df[COL_HH_INDEX])

df['_household_key'] = df['_community_clean'] + '|' + df['_address_clean']
df['_household_member_key'] = df['_household_key'] + '|' + df['_hh_index_clean']

# Check whether the within-household respondent index distinguishes household members.
household_sizes = df.groupby('_household_key').size()
member_key_counts = df.groupby(['_household_key', '_hh_index_clean']).size()
duplicate_member_keys = member_key_counts[member_key_counts > 1]

print(f"[HOUSEHOLD CHECK] Unique households by community + address: {df['_household_key'].nunique()}")
print(f"[HOUSEHOLD CHECK] Unique household-member keys after adding 本户第份: {df['_household_member_key'].nunique()}")
print(f"[HOUSEHOLD CHECK] Households with >=2 respondents: {(household_sizes >= 2).sum()}")
if len(duplicate_member_keys) > 0:
    print(
        f"[HOUSEHOLD WARNING] {len(duplicate_member_keys)} household-member keys are duplicated "
        f"({int(duplicate_member_keys.sum())} rows involved). Please check whether these are repeated records."
    )
    print(duplicate_member_keys.head(10))
else:
    print("[HOUSEHOLD CHECK] No duplicated 本户第份 within the same community + address household.")

age_map_dict = {'a':15, 'b':24, 'c':34.5, 'd':44.5, 'e':54.5, 'f':64.5, 'g':75}
df['age_num'] = df[COL_AGE].astype(str).str.strip().map(age_map_dict).fillna(45.0)

# 1. Mobility Calculation
daily_hours_nonfixed = np.zeros(N)
for k in range(5):
    if COL_TRIP_FREQ[k] in df.columns:
        fk = df[COL_TRIP_FREQ[k]].map(_map_freq).fillna(0).values
        dk = df[COL_TRIP_DUR[k]].map(_map_dur).fillna(0).values
        daily_hours_nonfixed += (fk * dk) / 7.0

daily_hours_fixed = np.zeros(N)
is_key_group = np.zeros(N, dtype=bool)
status_arr = df[COL_WORK_STATUS].values

for i in range(N):
    if _infer_employed(status_arr[i]) or _infer_student(status_arr[i]):
        daily_hours_fixed[i] = 8.0
        is_key_group[i] = True

mobility_index = np.clip(daily_hours_fixed + daily_hours_nonfixed, 0, 12.0)
virtual_contact_counts = df[COL_CONTACT_COUNT].map(_map_contact_count).fillna(0).values

# =============== Network Construction ===============
G = nx.Graph(); G.add_nodes_from(range(N))

# Layer A: Home
house2members = defaultdict(list)
for i in range(N): house2members[df.iloc[i]['_household_key']].append(i)
for members in house2members.values():
    m = len(members)
    if m <= 1: continue
    w = HH_CONTACT_HOURS / (m - 1)
    for a in range(m):
        for b in range(a+1, m):
            G.add_edge(members[a], members[b], weight=w, layer='home')

# Layer B: Random Community Connections
nodes_indices = np.arange(N)
for i in range(N):
    internal_social_time = daily_hours_nonfixed[i] * INTERNAL_SOCIAL_RATIO
    if internal_social_time > 0:
        n_encounters = int(round(internal_social_time / SOCIAL_ENCOUNTER_DURATION))
        if n_encounters > 0:
            tgts = np.random.choice(nodes_indices[nodes_indices!=i], size=min(n_encounters, N-1), replace=False)
            w = SOCIAL_ENCOUNTER_DURATION
            for t in tgts:
                if G.has_edge(i, t): G[i][t]['weight'] += w
                else: G.add_edge(i, t, weight=w, layer='community_random')

# =============== Strength Calculation ===============
strength_internal = np.array([G.degree(n, weight='weight') for n in G.nodes()])

# Virtual Strength
strength_vir_work = virtual_contact_counts * daily_hours_fixed
external_social_time = daily_hours_nonfixed * (1.0 - INTERNAL_SOCIAL_RATIO)
strength_vir_social = external_social_time * EXT_SOCIAL_CONTACT_DENSITY
strength_virtual = strength_vir_work + strength_vir_social

# Total Strength
total_node_strength = strength_internal + strength_virtual

# Neighbor Strength
avg_neighbor_strength = np.zeros(N)
for i in range(N):
    neighbors = list(G.neighbors(i))
    if len(neighbors) > 0:
        sum_s = sum(total_node_strength[nbr] for nbr in neighbors)
        avg_neighbor_strength[i] = sum_s / len(neighbors)

print(f"[NETWORK] Built. Avg Total Strength: {total_node_strength.mean():.2f}")

# =============== Calibration ===============
def calibrate_beta(target_r0, gamma, graph):
    adj_mat = nx.adjacency_matrix(graph, weight='weight')
    try:
        eigenvalues, _ = eigs(adj_mat, k=1, which='LM')
        sr = float(np.real(eigenvalues[0]))
    except Exception as exc:
        sr = 10.0
        print(
            f"[WARNING] Spectral-radius calculation failed ({exc}); "
            "using the legacy fallback value 10.0."
        )
    return (target_r0 * gamma * 24.0) / sr

if TARGET_R0:
    sum_internal = np.sum(strength_internal)

    work_exposure = np.sum(strength_vir_work[is_key_group]) * EXTERNAL_PREVALENCE
    social_exposure = np.sum(strength_vir_social[daily_hours_nonfixed > 0]) * EXTERNAL_PREVALENCE
    sum_external = work_exposure + social_exposure

    internal_ratio = sum_internal / (sum_internal + sum_external + 1e-9)

    ADJUSTED_TARGET_R0 = TARGET_R0 * internal_ratio
    
    print(f"[CALIBRATION] Internal exposure ratio: {internal_ratio:.2%}")
    print(f"[CALIBRATION] Adjusted Target R0 for network: {ADJUSTED_TARGET_R0:.4f}")

    BETA_EDGE = calibrate_beta(ADJUSTED_TARGET_R0, PROB_I_TO_S, G)
    print(f"[CALIBRATION] BETA_EDGE (Adjusted): {BETA_EDGE:.6f}")
else: 
    raise ValueError("BETA_EDGE not set!")

# =============== Simulation ===============
def simulate_once(seeds, days=SIMULATION_DAYS):
    state = np.zeros(N, dtype=int); state[seeds] = 2
    days_since_rec = np.full(N, -1, dtype=int)
    first_inf = np.full(N, np.inf); first_inf[seeds] = 0
    
    for d in range(1, days + 1):
        new_state = state.copy()
        r_val = np.random.random(N); r_rec = np.random.random(N)
        
        for i in range(N):
            if state[i] == 0: 
                lambda_int = 0.0
                for nbr, attr in G[i].items():
                    if state[nbr] == 2: lambda_int += BETA_EDGE * (attr['weight'] / 24.0)
                
                lambda_ext_work = 0.0
                if is_key_group[i]:
                    lambda_ext_work = BETA_EDGE * (strength_vir_work[i] / 24.0) * EXTERNAL_PREVALENCE
                
                lambda_ext_social = 0.0
                if daily_hours_nonfixed[i] > 0:
                    lambda_ext_social = BETA_EDGE * (strength_vir_social[i] / 24.0) * EXTERNAL_PREVALENCE

                lambda_total = lambda_int + lambda_ext_work + lambda_ext_social

                if 0 <= days_since_rec[i] <= 90:
                    nat_susc_mult = 1.0 - NAT_EFF_MAX
                elif 90 < days_since_rec[i] <= 365:
                    decay_ratio = (days_since_rec[i] - 90) / (365 - 90)
                    current_efficacy = NAT_EFF_MAX - (NAT_EFF_MAX - NAT_EFF_MIN) * decay_ratio
                    nat_susc_mult = 1.0 - current_efficacy
                elif days_since_rec[i] > 365:
                    nat_susc_mult = 1.0 - NAT_EFF_MIN
                else:
                    nat_susc_mult = 1.0

                prob = 1.0 - math.exp(-nat_susc_mult * lambda_total)
                
                if prob > 0 and r_val[i] < prob:
                    new_state[i] = 1; 
                    if math.isinf(first_inf[i]): first_inf[i] = d
            
            elif state[i] == 1: 
                if r_val[i] < PROB_E_TO_I: new_state[i] = 2
            elif state[i] == 2: 
                if r_rec[i] < PROB_I_TO_S:
                    new_state[i] = 0
                    days_since_rec[i] = 0
        
        mask_increment = (new_state == 0) & (days_since_rec >= 0)
        days_since_rec[mask_increment] += 1
        mask_infected = (new_state != 0)
        days_since_rec[mask_infected] = -1
        state = new_state
    return first_inf

print(f"[SIM] Running Monte Carlo ({MC_RUNS_TOTAL} runs)...")
all_first_times = []
for r in range(MC_RUNS_TOTAL):
    seeds = np.random.choice(N, size=NUM_SEEDS_PER_RUN, replace=False)
    all_first_times.append(simulate_once(seeds))
    if (r+1)%20==0: print(f" Run {r+1} done.")

first_time_mean = np.ma.masked_invalid(np.vstack(all_first_times)).mean(axis=0).filled(np.nan)

# =============== Analysis & Plotting ===============
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# RGB [178, 54, 36] / 255 = (0.698, 0.212, 0.141), hex #B23624
# RGB [110,151,177] / 255 = (0.431, 0.592, 0.694), hex #6E97B1
# Early-infected/high-risk group is shown in red; late-infected/reference group is shown in blue.
COLOR_EARLY = (178/255, 54/255, 36/255)    # red  #B23624
COLOR_LATE  = (110/255, 151/255, 177/255)  # blue #6E97B1

# Initialize Excel Writer for saving all plot data
excel_writer = pd.ExcelWriter(str(OUTPUT_DIR / "Plot_Data_Summary.xlsx"), engine='openpyxl')
output_excel_writer = pd.ExcelWriter(str(OUTPUT_DIR / "output_plot_data.xlsx"), engine='openpyxl')
violin_stats_records = []

# Assemble node-level predictors and the infection-timing response.
feat_df = pd.DataFrame({
    'node_id': np.arange(N),
    'first_time_mean': first_time_mean,
    'strength': total_node_strength, 
    'avg_neighbor_strength': avg_neighbor_strength,
    'age': df['age_num'].values,
    'mobility_index': mobility_index,
    'is_key_group': is_key_group.astype(int) # 0 or 1
}).dropna(subset=['first_time_mean']).reset_index(drop=True)

# Save foundational data for violin plots to excel
feat_df.to_excel(excel_writer, sheet_name='Violin_Base_Data', index=False)

# Violin Plots
THRESHOLDS = [0.10, 0.20, 0.50]
FEATURES_TO_PLOT = [
    ('strength', 'Total weighted exposure strength'),
    ('avg_neighbor_strength', 'Average neighbour exposure strength'),
    ('age', 'Age'),
    ('mobility_index', 'Daily activity duration')
]

def draw_significance_bracket(ax, x1, x2, y_max, p_val):
    h = y_max * 0.05; y = y_max + h
    star = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c='k')
    ax.text((x1+x2)*.5, y+h, star, ha='center', va='bottom', fontsize=11, fontweight='bold')

def plot_violin_for_feature(feature_col, feature_label):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)
    for ax, q in zip(axes, THRESHOLDS):
        cutoff = np.percentile(feat_df['first_time_mean'], int(q*100))
        early = feat_df[feat_df['first_time_mean'] <= cutoff][feature_col].dropna().values
        late  = feat_df[feat_df['first_time_mean'] > cutoff][feature_col].dropna().values
        if len(early)==0 or len(late)==0: continue
        
        data = [early, late]
        parts = ax.violinplot(data, positions=[1, 2], showmeans=False, showextrema=False, widths=0.7)
        parts['bodies'][0].set_facecolor(COLOR_EARLY); parts['bodies'][1].set_facecolor(COLOR_LATE)
        for pc in parts['bodies']: pc.set_alpha(0.75); pc.set_edgecolor('black')
        ax.boxplot(data, positions=[1, 2], widths=0.15, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor='white', alpha=0.8), medianprops=dict(color='black'))
        
        ax.set_title(f"Top {int(q*100)}% Earliest", fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks([1, 2]); ax.set_xticklabels(['Early Group', 'Late Group'])
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        
        try:
            _, p_val = stats.mannwhitneyu(
                early,
                late,
                alternative="two-sided",
                method="auto"
                )       
            y_max = max(max(early), max(late))
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
            violin_stats_records.append({
                'Feature': feature_col,
                'Threshold': f"Top {int(q*100)}%",
                'Early_Mean': np.mean(early),
                'Late_Mean': np.mean(late),
                'Early_Median': np.median(early),
                'Late_Median': np.median(late),
                'P_Value': p_val,
                'Significance': sig
            })
            draw_significance_bracket(ax, 1, 2, y_max, p_val)
            ax.set_ylim(top=y_max * 1.25)
        except (ValueError, TypeError) as exc:
            print(f"[WARNING] Mann--Whitney test skipped for {feature_col}: {exc}")
    axes[0].set_ylabel(feature_label, fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(str(OUTPUT_DIR / f"violin_{feature_col}_pub.png"), dpi=300, bbox_inches='tight')

# Compare the binary work/school activity indicator between timing groups.
def plot_violin_for_key_group():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, q in zip(axes, THRESHOLDS):
        cutoff = np.percentile(feat_df['first_time_mean'], int(q*100))
        early = feat_df[feat_df['first_time_mean'] <= cutoff]['is_key_group'].values
        late  = feat_df[feat_df['first_time_mean'] > cutoff]['is_key_group'].values
        if len(early)==0 or len(late)==0: continue
        
        data = [early, late]
        parts = ax.violinplot(data, positions=[1, 2], showmeans=False, showextrema=False, widths=0.7)
        parts['bodies'][0].set_facecolor(COLOR_EARLY); parts['bodies'][1].set_facecolor(COLOR_LATE)
        for pc in parts['bodies']: pc.set_alpha(0.75); pc.set_edgecolor('black')
        
        ax.boxplot(data, positions=[1, 2], widths=0.15, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor='white', alpha=0.8), medianprops=dict(color='black'))
        
        ax.set_title(f"Top {int(q*100)}% Earliest", fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks([1, 2]); ax.set_xticklabels(['Early', 'Late'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Non-Key (0)', 'Key Group (1)'])
        ax.set_ylim(-0.2, 1.4) 
        try:
            _, p_val = stats.mannwhitneyu(early, late)
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
            violin_stats_records.append({
                'Feature': 'is_key_group',
                'Threshold': f"Top {int(q*100)}%",
                'Early_Mean': np.mean(early),
                'Late_Mean': np.mean(late),
                'Early_Median': np.median(early),
                'Late_Median': np.median(late),
                'P_Value': p_val,
                'Significance': sig
            })
            draw_significance_bracket(ax, 1, 2, 1.1, p_val)
        except (ValueError, TypeError) as exc:
            print(f"[WARNING] Mann--Whitney test skipped for activity status: {exc}")
    axes[0].set_ylabel("Work/school activity status", fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(str(OUTPUT_DIR / "violin_key_group_pub.png"), dpi=300)

print("[PLOT] Generating Plots...")
if len(feat_df) > 50:
    for f_col, f_lbl in FEATURES_TO_PLOT: plot_violin_for_feature(f_col, f_lbl)
    plot_violin_for_key_group()

# ML Analysis
# Sex was removed from propagation-factor analysis because it showed negligible contribution.
# Retained predictors: total weighted node strength, average neighbor strength,
# age, mobility/activity intensity, and key activity-group status.
X_cols = ['strength', 'avg_neighbor_strength', 'age', 'mobility_index', 'is_key_group']
X = feat_df[X_cols]; y = feat_df['first_time_mean']

rename_dict = {
    'strength': 'Total weighted exposure strength',
    'avg_neighbor_strength': 'Average neighbour exposure strength',
    'age': 'Age',
    'mobility_index': 'Daily activity duration',
    'is_key_group': 'Work/school activity status',
    'First_Time': 'Mean first infection time'
}

if len(X) > 20:
    print("\n[ML] Starting infection-timing predictor analysis...")

    # -------------------------------------------------------
    # 0. Data Prep
    # -------------------------------------------------------
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize (for Lasso/Logistic)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_cols)

    # -------------------------------------------------------
    # -------------------------------------------------------
    plt.figure(figsize=(10, 8))
    # Create a display version with nice names
    plot_df_display = pd.concat([X, y.rename("First_Time")], axis=1).rename(columns=rename_dict)
    
    # Save Pearson correlation data to excel
    pearson_corr_matrix = plot_df_display.corr()
    pearson_corr_matrix.to_excel(excel_writer, sheet_name='Pearson_Correlation_Matrix')
    pearson_target = pearson_corr_matrix['Mean first infection time'].drop('Mean first infection time')
    
    sns.heatmap(pearson_corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', center=0, cbar_kws={"shrink": .8})
    plt.title("Pearson correlation matrix", fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "analysis_1_pearson_heatmap.png"), dpi=300)
    print("  > [1/7] Saved: analysis_1_pearson_heatmap.png")

    # -------------------------------------------------------
    # -------------------------------------------------------
    lasso_model = LassoCV(cv=5, random_state=42).fit(X_train_scaled, y_train)
    r2_lasso = r2_score(y_test, lasso_model.predict(X_test_scaled))
    
    # Plot Coefficients
    lasso_res = pd.Series(lasso_model.coef_, index=X_cols).rename(index=rename_dict)
    
    # Save Lasso data to excel
    lasso_res.to_excel(excel_writer, sheet_name='Lasso_Coefficients')
    
    lasso_res_filtered = lasso_res[lasso_res != 0].sort_values(key=abs, ascending=True)
    
    plt.figure(figsize=(8, 6))
    if not lasso_res_filtered.empty:
        colors = ['#d62728' if x < 0 else '#1f77b4' for x in lasso_res_filtered]
        lasso_res_filtered.plot(kind='barh', color=colors)
        plt.title(f"Lasso Coefficients (R²={r2_lasso:.2f})", fontweight='bold')
        plt.xlabel("Standardized Coefficient Value")
        plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
    else:
        plt.text(0.5, 0.5, "Lasso shrunk all coefs to 0", ha='center')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "analysis_2_lasso_importance.png"), dpi=300)
    print("  > [2/7] Saved: analysis_2_lasso_importance.png")

    # -------------------------------------------------------
    # -------------------------------------------------------
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
    r2_rf = r2_score(y_test, rf_model.predict(X_test))
    
    rf_res = pd.Series(rf_model.feature_importances_, index=X_cols).rename(index=rename_dict)
    
    # Save RF data to excel
    rf_res.to_excel(excel_writer, sheet_name='RandomForest_Importance')
    pd.DataFrame({'Importance_Score': rf_res.sort_values(ascending=True)}).to_excel(output_excel_writer, sheet_name='RF_Importance')
    
    rf_res_sorted = rf_res.sort_values(ascending=True)
    
    plt.figure(figsize=(8, 6))
    rf_res_sorted.plot(kind='barh', color='#2ca02c')
    plt.title(f"Random Forest Importance (R²={r2_rf:.2f})", fontweight='bold')
    plt.xlabel("Importance Score (MDI)")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "analysis_3_rf_importance.png"), dpi=300)
    print("  > [3/7] Saved: analysis_3_rf_importance.png")

    # -------------------------------------------------------
    # -------------------------------------------------------
    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X)
        X_display = X.rename(columns=rename_dict)
        
        # Calculate mean absolute SHAP for tabular output & save to excel
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_res = pd.Series(mean_abs_shap, index=X_cols).rename(index=rename_dict)
        shap_res.to_excel(excel_writer, sheet_name='SHAP_Mean_Abs_Importance')
        
        plt.figure()
        shap.summary_plot(shap_values, X_display, show=False)
        plt.title("SHAP Feature Importance (Impact Direction)", fontweight='bold', pad=30)
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "analysis_4_shap_summary.png"), dpi=300, bbox_inches='tight')
        print("  > [4/7] Saved: analysis_4_shap_summary.png")
    else:
        shap_res = pd.Series(0, index=X_cols).rename(index=rename_dict)

    # -------------------------------------------------------
    # -------------------------------------------------------
    df_rf_shap = pd.DataFrame({
        'Random forest': rf_res,
        'SHAP': shap_res
    }).fillna(0)

    df_lasso_pearson = pd.DataFrame({
        r'Pearson $|r|$': pearson_target.abs(),
        r'Lasso $|\beta|$': lasso_res.abs()
    }).fillna(0)

    df_rf_shap.to_excel(excel_writer, sheet_name='Summary_RF_and_SHAP')
    df_lasso_pearson.to_excel(excel_writer, sheet_name='Summary_Lasso_and_Pearson')

    def plot_beautiful_heatmap(df, title, filename):
        column_max = df.abs().max(axis=0).replace(0, np.nan)
        df_norm = df.abs().div(column_max, axis=1).fillna(0)
        
        plt.figure(figsize=(7, len(df)*0.8 + 1.5))
        sns.heatmap(df_norm, annot=True, fmt=".2f", cmap='Greens', cbar=True,
                    linewidths=1.5, linecolor='white', annot_kws={"size": 13})
        
        plt.title(title, fontweight='bold', fontsize=14, pad=15)
        plt.yticks(rotation=0, fontsize=12)
        plt.xticks(rotation=90, fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    plot_beautiful_heatmap(df_rf_shap, "Normalized feature importance: random forest and SHAP", str(OUTPUT_DIR / "analysis_table_1_rf_shap.png"))
    plot_beautiful_heatmap(df_lasso_pearson, "Normalized linear benchmark magnitudes: Pearson and Lasso", str(OUTPUT_DIR / "analysis_table_2_lasso_pearson.png"))
    print("  > [Split Tables] Saved (Heatmap Style): analysis_table_1_rf_shap.png & analysis_table_2_lasso_pearson.png")

    # -------------------------------------------------------
    # -------------------------------------------------------
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Lasso (Linear)', 'Random Forest (Non-linear)'], [r2_lasso, r2_rf], 
                   color=['#4DBBD5', '#E64B35'], alpha=0.8, edgecolor='k')
    plt.ylabel('R² Score (Goodness of Fit)')
    plt.title('Model Predictive Power Comparison', fontweight='bold')
    plt.ylim(0, max(0.1, max(r2_lasso, r2_rf) * 1.25))
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "analysis_5_r2_comparison.png"), dpi=300)
    print("  > [5/7] Saved: analysis_5_r2_comparison.png")

    # ==========================================
    # 6. Exploratory classification of early-infected nodes
    # ==========================================
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    X_train_roc, X_test_roc, y_train_roc, y_test_roc = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    threshold_val = np.percentile(y_train_roc, 25)
    
    y_train_bin = (y_train_roc <= threshold_val).astype(int)
    y_test_bin = (y_test_roc <= threshold_val).astype(int)
    
    clf_log = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    clf_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
    
    scaler_bin = StandardScaler()
    X_train_roc_scaled = scaler_bin.fit_transform(X_train_roc)
    X_test_roc_scaled = scaler_bin.transform(X_test_roc)
    
    clf_log.fit(X_train_roc_scaled, y_train_bin)
    clf_rf.fit(X_train_roc_scaled, y_train_bin) 
    
    prob_log = clf_log.predict_proba(X_test_roc_scaled)[:, 1]
    prob_rf = clf_rf.predict_proba(X_test_roc_scaled)[:, 1]
    
    fpr_log, tpr_log, _ = roc_curve(y_test_bin, prob_log)
    auc_log = auc(fpr_log, tpr_log)
    
    fpr_rf, tpr_rf, _ = roc_curve(y_test_bin, prob_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    # Save ROC Data to excel
    roc_log_df = pd.DataFrame({'FPR': fpr_log, 'TPR': tpr_log})
    roc_rf_df = pd.DataFrame({'FPR': fpr_rf, 'TPR': tpr_rf})
    roc_log_df.to_excel(excel_writer, sheet_name='ROC_Logistic_Data', index=False)
    roc_rf_df.to_excel(excel_writer, sheet_name='ROC_RF_Data', index=False)
    roc_log_df.to_excel(output_excel_writer, sheet_name='ROC_Logistic', index=False)
    roc_rf_df.to_excel(output_excel_writer, sheet_name='ROC_RandomForest', index=False)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc_log:.3f})', 
             color='#1f77b4', linewidth=2.5)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})', 
             color='#2ca02c', linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)', alpha=0.7)
    
    plt.title('ROC Curve for High Risk Identification', fontweight='bold', fontsize=15, pad=15)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'ROC_Curve_Corrected.png'), dpi=300)
    
    print(f"  > [6/7] Strict AUC (Logistic Regression): {auc_log:.4f}")
    print(f"  > [6/7] Strict AUC (Random Forest): {auc_rf:.4f}")

# Close and save the Excel file
pd.DataFrame(violin_stats_records).to_excel(output_excel_writer, sheet_name='Violin_Stats', index=False)
excel_writer.close()
output_excel_writer.close()
print(f"\n[SUCCESS] Plot data saved to {OUTPUT_DIR / 'Plot_Data_Summary.xlsx'}.")