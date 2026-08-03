"""Run the primary vaccination-strategy simulation experiment.

The script constructs the survey-derived weighted exposure network, calibrates
transmission intensity, evaluates vaccination strategies across the primary
parameter grid, and retains the summary and replicate-level data used by the
main and supplementary figures.
"""
import os
import sys
from pathlib import Path

# Limit implicit numerical-library threading to avoid nested parallelism.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')

import re
import math
import random
import itertools
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from scipy.sparse.linalg import eigsh

# =============== Parameters ===============
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_MODE = '--test-mode' in sys.argv
DATA_FILE = '20231012.xlsx'
DATA_PATH = str(PROJECT_ROOT / 'data' / DATA_FILE)
SHEET_ID = 0

# Parallel execution settings
MAX_CORES = 22

# ---- SEIS Parameters ----
PROB_E_TO_I = 0.2
PROB_I_TO_S = 0.142

# Piecewise natural-immunity efficacy
NAT_EFF_MAX = 0.652
NAT_EFF_MIN = 0.247

# VES_0_LIST = [0.50]
# R0_LIST = [7.0]
# MC_RUNS_PER_POINT = 10  

# ---- Network & Time Parameters ----
HH_CONTACT_HOURS = 12.0 
SOCIAL_ENCOUNTER_DURATION = 0.25 
INTERNAL_SOCIAL_RATIO = 0.3   # Fraction of social activity allocated within the community

# ---- External Virtual Parameters ----
VIRTUAL_CONTACT_MAPPING = {'a': 3.0, 'b': 10.0, 'c': 20.0, 'd': 30.0}
EXTERNAL_PREVALENCE = 0.005   # Background prevalence outside the modelled community
EXT_SOCIAL_CONTACT_DENSITY = 4.0 
NUM_SEEDS_PER_RUN = 10
RNG_SEED = 42

# Baseline vaccine-efficacy scenarios
VES_0_LIST = [0.20, 0.35, 0.50, 0.65, 0.80]

# Parameter-grid settings
R0_LIST = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.3, 3.5, 4.0, 5.0, 7.0, 8.5, 10.0, 12.0, 14.0, 16.0, 18.6]

MC_RUNS_PER_POINT = 100       
SIM_DAYS_STRATEGY = 500     
COVERAGE_STEPS = np.linspace(0, 0.95, 20) 

# A lightweight synthetic-data smoke test can be enabled from the command line.
# The test exercises the production simulation path but is not intended to
# reproduce any numerical result reported in the manuscript.
if TEST_MODE:
    MAX_CORES = 1
    NUM_SEEDS_PER_RUN = 3
    VES_0_LIST = [0.50]
    R0_LIST = [2.0]
    MC_RUNS_PER_POINT = 2
    SIM_DAYS_STRATEGY = 30
    COVERAGE_STEPS = np.array([0.0, 0.5], dtype=float)

# =============== Data-retention settings ===============
# This script does not generate figures. It retains the simulation outputs needed
# for the main-text figures and the planned Supplementary Results analyses.
OUTPUT_ROOT = str(
    PROJECT_ROOT / 'results' / ('smoke_test' if TEST_MODE else 'primary_simulation')
)
SCENARIO_SUMMARY_DIR = os.path.join(OUTPUT_ROOT, "scenario_summaries")
FIG4_BIN_DIR = os.path.join(OUTPUT_ROOT, "fig4_exact_bin_counts")
SAMPLED_FEATURE_DIR = os.path.join(OUTPUT_ROOT, "sampled_node_distributions")
FEATURE_SUMMARY_DIR = os.path.join(OUTPUT_ROOT, "exact_feature_summaries")
UNCERTAINTY_REPLICATE_DIR = os.path.join(OUTPUT_ROOT, "selected_uncertainty_replicates")
METADATA_DIR = os.path.join(OUTPUT_ROOT, "metadata")

for _output_dir in [
    OUTPUT_ROOT,
    SCENARIO_SUMMARY_DIR,
    FIG4_BIN_DIR,
    SAMPLED_FEATURE_DIR,
    FEATURE_SUMMARY_DIR,
    UNCERTAINTY_REPLICATE_DIR,
    METADATA_DIR
]:
    os.makedirs(_output_dir, exist_ok=True)

# Exact Fig. 4-style exposure-bin counts are retained for these two strategies
# across ALL R0, VES_0 and coverage combinations. At 20%, 50% and 80% coverage,
# the same bin-level data are additionally retained for every strategy.
FIG4_BIN_STRATEGIES = {"Random", "High Strength (Node)"}
DETAILED_FEATURE_COVERAGES = [0.20, 0.50, 0.80]

# Replicate-level scalar outcomes are retained only for scenarios selected a priori
# from the main-text findings. All three principal strategies are retained so that
# uncertainty can be evaluated without treating MC run numbers as paired samples.
UNCERTAINTY_SELECTED_SCENARIOS = [
    # Strong high-exposure peak-suppression benefit in the main text
    {"VES_0": 0.50, "R0": 2.0,  "Coverage": 0.50, "Label": "HighExposure_Peak_StrongBenefit"},
    # Strongest main-text reversal in final infection proportion; also used in Fig. 4
    {"VES_0": 0.80, "R0": 4.0,  "Coverage": 0.50, "Label": "HighExposure_FinalInfection_Reversal"},
    # Strong oldest-first mortality benefit in the main text
    {"VES_0": 0.80, "R0": 1.2,  "Coverage": 0.50, "Label": "OldestFirst_Deaths_StrongBenefit"},
    # Strong oldest-first disadvantage in epidemic peak size
    {"VES_0": 0.80, "R0": 1.5,  "Coverage": 0.50, "Label": "OldestFirst_Peak_StrongDisadvantage"},
    # Near-zero high-exposure peak difference at the upper end of transmissibility
    {"VES_0": 0.50, "R0": 18.6, "Coverage": 0.50, "Label": "HighExposure_Peak_NearZero"}
]
UNCERTAINTY_STRATEGIES = {
    "Random",
    "High Strength (Node)",
    "Priority: Oldest First"
}

# Retain the original stratified samples for exploratory violin plots.
# They are visualization samples only and must not be used to estimate state
# proportions, bin weights or inferential significance.
SAVE_SAMPLED_NODE_DISTRIBUTIONS = False
SAMPLED_N_PER_STATUS = 2000

np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

# =============== IFR Data (The Lancet 2020) ===============
IFR_MAP = {
    15:   0.0000695, 
    24:   0.000309,  
    34.5: 0.000844,  
    44.5: 0.00161,   
    54.5: 0.00595,   
    64.5: 0.0193,    
    75:   0.0428     
}

# =============== Helper Functions ===============
def _normalize_col(s): return str(s).replace('\ufeff', '').strip().replace(' ', '').replace('（', '(').replace('）', ')').replace('/', '')
def pick_column(df, targets):
    norm_map = {col: _normalize_col(col) for col in df.columns}
    wanted_norm = [_normalize_col(t) for t in targets]
    for col, ncol in norm_map.items():
        if ncol in wanted_norm: return col
    for col, ncol in norm_map.items():
        if any(w in ncol for w in wanted_norm) or any(ncol in w for w in wanted_norm): return col
    raise KeyError(f"Column not found: {targets}")

FREQ_MAP = {'无': 0, '<1次': 0.5, '1-2次': 1.5, '3-4次': 3.5, '≥5次': 5.5, 'a': 5.5, 'b': 3.5, 'c': 1.5, 'd': 0.5}
DUR_MAP = {'无': 0, '2小时以内': 1, '2-5小时': 3.5, '5小时以上': 6, 'a': 1, 'b': 3.5, 'c': 6}
def _map_freq(x): s=str(x).strip(); return FREQ_MAP.get(s) if s in FREQ_MAP else (5.5 if '5次' in s else 0.5)
def _map_dur(x): s=str(x).strip(); return DUR_MAP.get(s) if s in DUR_MAP else (3.5 if '2-5' in s else 0.0)
def _infer_employed(s): s=str(s).strip(); return s in ['b', 'B', 'd', 'D'] or '工' in s or '班' in s
def _infer_student(s): s=str(s).strip(); return '学' in s or s in ['a', 'A']
def _map_contact_count(x):
    s = str(x).lower().strip(); 
    for k,v in VIRTUAL_CONTACT_MAPPING.items(): 
        if k in s: return v
    return 0.0 

def calculate_expected_deaths(infected_indices, age_series):
    deaths = 0.0
    ages = age_series.iloc[infected_indices].values
    available_ages = np.array(list(IFR_MAP.keys()))
    for age in ages:
        idx = (np.abs(available_ages - age)).argmin()
        closest_age = available_ages[idx]
        deaths += IFR_MAP[closest_age]
    return deaths

# =============== Data Loading ===============
def build_synthetic_survey_dataframe(number_of_nodes=60, household_size=3):
    """Create a deterministic questionnaire-shaped dataset for smoke testing.

    The generated records contain no participant data. They only reproduce the
    column schema and categorical encodings required by the production script.
    """
    if number_of_nodes < 12:
        raise ValueError("The synthetic smoke test requires at least 12 nodes.")
    if household_size < 2:
        raise ValueError("household_size must be at least 2.")

    age_codes = ["a", "b", "c", "d", "e", "f", "g"]
    work_codes = ["a", "b", "c", "d"]
    contact_codes = ["a", "b", "c", "d"]
    frequency_codes = ["无", "<1次", "1-2次", "3-4次", "≥5次"]
    duration_codes = ["无", "2小时以内", "2-5小时", "5小时以上"]

    records = []
    for node_id in range(number_of_nodes):
        household_id = node_id // household_size
        record = {
            "小区名称": f"Synthetic_Community_{household_id % 3 + 1}",
            "住户详细地址": f"Synthetic_Household_{household_id + 1:03d}",
            "本户第份": node_id % household_size + 1,
            "工作状态": work_codes[node_id % len(work_codes)],
            "人员数量": contact_codes[node_id % len(contact_codes)],
            "年龄": age_codes[node_id % len(age_codes)],
        }
        for trip_index in range(1, 6):
            offset = node_id + trip_index
            record[f"频率{trip_index}"] = frequency_codes[offset % len(frequency_codes)]
            record[f"时长{trip_index}"] = duration_codes[offset % len(duration_codes)]
        records.append(record)

    return pd.DataFrame.from_records(records)


if TEST_MODE:
    df = build_synthetic_survey_dataframe()
    print(
        "[TEST MODE] Using deterministic synthetic questionnaire-shaped data. "
        "The resulting outputs are execution checks only and must not be "
        "interpreted as manuscript results.",
        flush=True,
    )
else:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
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

# Household definition:
df['_community_clean'] = df[COL_COMMUNITY].astype(str).str.strip()
df['_address_clean'] = df[COL_ADDRESS].astype(str).str.strip()
df['_hh_index_clean'] = df[COL_HH_INDEX].astype(str).str.strip()
df['_household_key'] = df['_community_clean'] + '|' + df['_address_clean']
df['_household_member_key'] = df['_household_key'] + '|' + df['_hh_index_clean']

household_sizes = df.groupby('_household_key').size()
duplicated_member_keys = df['_household_member_key'].duplicated(keep=False)
print(f"[HOUSEHOLD CHECK] Unique households by community + address: {df['_household_key'].nunique()}")
print(f"[HOUSEHOLD CHECK] Unique household-member keys after adding 本户第份: {df['_household_member_key'].nunique()}")
print(f"[HOUSEHOLD CHECK] Households with >=2 respondents: {(household_sizes >= 2).sum()}")
if duplicated_member_keys.any():
    print(f"[HOUSEHOLD WARNING] {duplicated_member_keys.sum()} rows have duplicated 小区名称 + 住户详细地址 + 本户第份 keys. Please check whether these are duplicate records or household-index coding issues.")

age_map_dict = {'a':15, 'b':24, 'c':34.5, 'd':44.5, 'e':54.5, 'f':64.5, 'g':75}
df['age_num'] = df[COL_AGE].astype(str).str.strip().map(age_map_dict).fillna(45.0)

# Mobility & Key Group
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
house2members = defaultdict(list)
for i in range(N): house2members[df.iloc[i]['_household_key']].append(i)
for members in house2members.values():
    m = len(members); 
    if m <= 1: continue
    w = HH_CONTACT_HOURS / (m - 1)
    for a in range(m):
        for b in range(a+1, m): G.add_edge(members[a], members[b], weight=w, layer='home')

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

strength_internal = np.array([G.degree(n, weight='weight') for n in G.nodes()])
strength_vir_work = virtual_contact_counts * daily_hours_fixed
external_social_time = daily_hours_nonfixed * (1.0 - INTERNAL_SOCIAL_RATIO)
strength_vir_social = external_social_time * EXT_SOCIAL_CONTACT_DENSITY
strength_virtual = strength_vir_work + strength_vir_social
total_node_strength = strength_internal + strength_virtual

# =============== Exact bin-level data retention for Fig. 4 ===============
FEATURE_BIN_COUNT = 12
positive_strength_mask = total_node_strength > 0
positive_strength_values = total_node_strength[positive_strength_mask]

if len(positive_strength_values) == 0:
    raise ValueError("No positive total_node_strength values are available for logarithmic binning.")

feature_strength_bins = np.logspace(
    np.log10(positive_strength_values.min()),
    np.log10(positive_strength_values.max()),
    FEATURE_BIN_COUNT + 1
)

feature_strength_bin_ids = np.full(N, -1, dtype=int)
feature_strength_bin_ids[positive_strength_mask] = pd.cut(
    total_node_strength[positive_strength_mask],
    bins=feature_strength_bins,
    labels=False,
    include_lowest=True
).astype(int)

# Precompute node indices and metadata for each bin once. This reduces the
# additional cost of retaining exact bin-level counts during Monte Carlo runs.
feature_bin_definitions = []
feature_bin_node_indices = {}
for _feature_bin_id in [-1] + list(range(FEATURE_BIN_COUNT)):
    _node_indices = np.where(feature_strength_bin_ids == _feature_bin_id)[0]
    feature_bin_node_indices[_feature_bin_id] = _node_indices

    if _feature_bin_id == -1:
        _saved_bin_id = 0
        _bin_left = 0.0
        _bin_right = 0.0
        _bin_center = 0.0
        _bin_type = "Zero/non-positive strength"
    else:
        _saved_bin_id = _feature_bin_id + 1
        _bin_left = float(feature_strength_bins[_feature_bin_id])
        _bin_right = float(feature_strength_bins[_feature_bin_id + 1])
        _bin_center = float(np.sqrt(_bin_left * _bin_right))
        _bin_type = "Positive log-spaced bin"

    feature_bin_definitions.append({
        "Internal_Bin_ID": _feature_bin_id,
        "Bin_ID": _saved_bin_id,
        "Bin_Type": _bin_type,
        "Bin_Left": _bin_left,
        "Bin_Right": _bin_right,
        "Bin_Center": _bin_center,
        "Total_Nodes": int(len(_node_indices)),
        "Node_Weight": float(len(_node_indices) / N)
    })

avg_neighbor_strength = np.zeros(N)
for i in range(N):
    neighbors = list(G.neighbors(i))
    if len(neighbors) > 0:
        sum_s = sum(total_node_strength[nbr] for nbr in neighbors)
        avg_neighbor_strength[i] = sum_s / len(neighbors)

print(f"[NETWORK] Built. Avg Total Strength: {total_node_strength.mean():.2f}")

def build_sparse_adj_matrix(graph):
    """
    Build a weighted sparse adjacency matrix compatible with both
    old and new NetworkX versions.
    """
    if hasattr(nx, "to_scipy_sparse_array"):
        return nx.to_scipy_sparse_array(
            graph,
            nodelist=range(N),
            weight='weight',
            format='csr',
            dtype=float
        )
    else:
        return nx.to_scipy_sparse_matrix(
            graph,
            nodelist=range(N),
            weight='weight',
            format='csr',
            dtype=float
        )

# ============================================
# ============================================
adj_matrix = build_sparse_adj_matrix(G)


# =============== Calibration Function ===============
def calibrate_beta(target_r0, gamma, graph):
    adj_mat = build_sparse_adj_matrix(graph)

    try:
        eigenvalues, _ = eigsh(adj_mat, k=1, which='LA')
        sr = float(eigenvalues[0])
        if sr <= 0:
            raise ValueError(f"Non-positive spectral radius: {sr}")
    except Exception as e:
        print(f"[WARNING] Eigenvalue calculation failed: {e}", flush=True)
        sr = 10.0

    return (target_r0 * gamma * 24.0) / sr

# ============================================
# Strategy & Simulation Functions
# ============================================
def get_vaccinated_mask(strategy, coverage, N):
    num_vax = int(coverage * N)
    if num_vax == 0: return np.zeros(N, dtype=bool)
    indices = []
    
    if strategy == 'Random': 
        indices = np.random.choice(N, num_vax, replace=False)

    elif strategy == 'Priority: Oldest First': 
        indices = np.argsort(df['age_num'].values)[::-1][:num_vax]
        
    elif strategy == 'Priority: Youngest First': 
        indices = np.argsort(df['age_num'].values)[:num_vax]

    elif strategy == 'Priority: Key Groups First':
        key_indices = np.where(is_key_group)[0]
        other_indices = np.where(~is_key_group)[0] 
        np.random.shuffle(key_indices)
        np.random.shuffle(other_indices)
        indices = np.concatenate([key_indices, other_indices])[:num_vax]

    elif strategy == 'High Strength (Node)': 
        indices = np.argsort(total_node_strength)[::-1][:num_vax]
        
    elif strategy == 'High Strength (Neighbor)': 
        indices = np.argsort(avg_neighbor_strength)[::-1][:num_vax]
        
    elif strategy == 'High Mobility (Duration)': 
        indices = np.argsort(mobility_index)[::-1][:num_vax]
        
    elif strategy == 'Household Priority: Hub-based':
        sorted_nodes = np.argsort(total_node_strength)[::-1]
        vax_set = set()
        household_arr = df['_household_key'].values
        
        hh2members = defaultdict(list)
        for i, hh in enumerate(household_arr):
            hh2members[hh].append(i)
            
        for node in sorted_nodes:
            if len(vax_set) >= num_vax:
                break
            
            hh = household_arr[node]
            members = hh2members[hh]
            
            if members[0] not in vax_set: 
                current_len = len(vax_set)
                added_len = current_len + len(members)
                
                if added_len > num_vax:
                    if abs(added_len - num_vax) <= abs(num_vax - current_len):
                        vax_set.update(members)
                    break 
                else:
                    vax_set.update(members)
                    
        indices = list(vax_set)
     
    elif strategy == 'Household Priority: Hub + Elderly':
        sorted_nodes = np.argsort(total_node_strength)[::-1]
        vax_set = set()
        household_arr = df['_household_key'].values
        ages_arr = df['age_num'].values
        
        hh2members = defaultdict(list)
        for i, hh in enumerate(household_arr):
            hh2members[hh].append(i)
            
        for node in sorted_nodes:
            if len(vax_set) >= num_vax:
                break
            
            if node not in vax_set:
                hh = household_arr[node]
                members = hh2members[hh]
                
                target_members = [m for m in members if m == node or ages_arr[m] >= 54.5]
                new_members = [m for m in target_members if m not in vax_set]
                
                if not new_members:
                    continue
                    
                current_len = len(vax_set)
                added_len = current_len + len(new_members)
                
                if added_len > num_vax:
                    if abs(added_len - num_vax) <= abs(num_vax - current_len):
                        vax_set.update(new_members)
                    break 
                else:
                    vax_set.update(new_members)
                    
        indices = list(vax_set)

    elif strategy == 'Household Priority: Elderly + Key Group':
        sorted_nodes = np.argsort(df['age_num'].values)[::-1]
        vax_set = set()
        household_arr = df['_household_key'].values
        
        hh2members = defaultdict(list)
        for i, hh in enumerate(household_arr):
            hh2members[hh].append(i)
            
        for node in sorted_nodes:
            if len(vax_set) >= num_vax:
                break
                
            if node not in vax_set:
                hh = household_arr[node]
                members = hh2members[hh]
                
                target_members = [m for m in members if m == node or is_key_group[m]]
                new_members = [m for m in target_members if m not in vax_set]
                
                if not new_members:
                    continue
                    
                current_len = len(vax_set)
                added_len = current_len + len(new_members)
                
                if added_len > num_vax:
                    if abs(added_len - num_vax) <= abs(num_vax - current_len):
                        vax_set.update(new_members)
                    break 
                else:
                    vax_set.update(new_members)
                    
        indices = list(vax_set)    
    
    mask = np.zeros(N, dtype=bool)
    mask[indices] = True
    return mask

def run_strategy_simulation(beta_val, vax_mask, ves_0):
    state = np.zeros(N, dtype=int)
    days_since_rec = np.full(N, -1, dtype=int) 
    ever_infected = np.zeros(N, dtype=bool)
    
    cumulative_exposure = np.zeros(N, dtype=float)
    
    lambda_external_const = (
        np.where(
            is_key_group,
            (beta_val * strength_vir_work / 24.0) * EXTERNAL_PREVALENCE,
            0.0
        )
        + np.where(
            daily_hours_nonfixed > 0,
            (beta_val * strength_vir_social / 24.0) * EXTERNAL_PREVALENCE,
            0.0
        )
    )
    
    seeds = np.random.choice(N, size=NUM_SEEDS_PER_RUN, replace=False)
    state[seeds] = 2; ever_infected[seeds] = True
    daily_infected_counts = []
    
    for d in range(1, SIM_DAYS_STRATEGY + 1):
        new_state = state.copy(); r_val = np.random.random(N); r_rec = np.random.random(N)
        
        # ==========================================
        # ==========================================
        if d <= 100:     
            current_ve = ves_0
        elif d <= 180:   
            current_ve = 0.7 * ves_0
        else:            
            current_ve = 0.4 * ves_0
            
        vax_susc_mult = np.where(vax_mask, 1.0 - current_ve, 1.0)
        
        # ==========================================
        # ==========================================
        nat_susc_mult = np.ones(N)
        
        mask_0_90 = (days_since_rec >= 0) & (days_since_rec <= 90)
        mask_91_365 = (days_since_rec > 90) & (days_since_rec <= 365)
        mask_gt_365 = (days_since_rec > 365)
        
        nat_susc_mult[mask_0_90] = 1.0 - NAT_EFF_MAX
        
        if np.any(mask_91_365):
            decay_ratio = (days_since_rec[mask_91_365] - 90) / (365 - 90)
            current_efficacy = NAT_EFF_MAX - (NAT_EFF_MAX - NAT_EFF_MIN) * decay_ratio
            nat_susc_mult[mask_91_365] = 1.0 - current_efficacy
            
        nat_susc_mult[mask_gt_365] = 1.0 - NAT_EFF_MIN
        
        # ==========================================
        # ==========================================
        total_susc_mult = vax_susc_mult * nat_susc_mult
        
        sus_indices = np.where(state == 0)[0]
        
        if len(sus_indices) > 0:
            I_vector = (state == 2).astype(float)
            
            lambda_internal = (beta_val / 24.0) * adj_matrix.dot(I_vector)
            
            lambda_total_all = lambda_internal + lambda_external_const
            lambda_sus = lambda_total_all[sus_indices]
            susc_sus = total_susc_mult[sus_indices]
            
            cumulative_exposure[sus_indices] += lambda_sus
            
            prob_sus = 1.0 - np.exp(-susc_sus * lambda_sus)
            infected_mask = (prob_sus > 0) & (r_val[sus_indices] < prob_sus)
            
            newly_infected_indices = sus_indices[infected_mask]
            new_state[newly_infected_indices] = 1
            ever_infected[newly_infected_indices] = True
        
        exp_indices = np.where(state == 1)[0]
        new_state[exp_indices] = np.where(r_val[exp_indices] < PROB_E_TO_I, 2, 1)
        
        inf_indices = np.where(state == 2)[0]
        recovered = r_rec[inf_indices] < PROB_I_TO_S
        new_state[inf_indices[recovered]] = 0
        
        days_since_rec[inf_indices[recovered]] = 0
        mask_increment = (new_state == 0) & (days_since_rec >= 0)
        days_since_rec[mask_increment] += 1
        mask_infected = (new_state != 0)
        days_since_rec[mask_infected] = -1
        
        state = new_state
        daily_infected_counts.append(np.sum(state == 2))
    
    unique_infected_total = np.sum(ever_infected)
    daily_infected_counts = np.array(daily_infected_counts)
    peak_sz = np.max(daily_infected_counts) if len(daily_infected_counts) > 0 else 0
    peak_tm = np.argmax(daily_infected_counts) if len(daily_infected_counts) > 0 else 0
    infected_indices = np.where(ever_infected)[0]
    expected_deaths = calculate_expected_deaths(infected_indices, df['age_num'])
    
    return unique_infected_total, peak_sz, peak_tm, expected_deaths, daily_infected_counts, ever_infected, cumulative_exposure

# ============================================
# MAIN EXECUTION LOOP (Multi-Scenario & Multi-R0)
# ============================================

strategies_list = [
    'Random', 
    'High Strength (Node)',          
    'High Strength (Neighbor)',      
    'High Mobility (Duration)',      
    'Priority: Oldest First',       
    'Priority: Youngest First',     
    'Priority: Key Groups First', 
    'Household Priority: Hub-based',
    'Household Priority: Hub + Elderly',
    'Household Priority: Elderly + Key Group'
]

# Principal strategies used in the main text and in the selected uncertainty analyses
main_strategies = [
    'Random',
    'High Strength (Node)',
    'Priority: Oldest First'
]

print(f"\n[INFO] Starting Multi-Scenario Simulation Loop")


def _match_selected_uncertainty_setting(current_ves0, target_r0, coverage):
    """
    Return the pre-specified uncertainty-analysis label and data role.

    For each selected target setting, replicate-level outcomes are retained at
    the target coverage. The corresponding Random, coverage=0 simulations are
    also retained as the common no-vaccination baseline Y0.
    """
    for item in UNCERTAINTY_SELECTED_SCENARIOS:
        same_parameter_setting = (
            np.isclose(current_ves0, item["VES_0"], atol=1e-12)
            and np.isclose(target_r0, item["R0"], atol=1e-12)
        )
        if not same_parameter_setting:
            continue

        if np.isclose(coverage, item["Coverage"], atol=0.01):
            return item["Label"], "Target coverage"
        if np.isclose(coverage, 0.0, atol=1e-12):
            return item["Label"], "No-vaccination baseline"

    return None


def _summarize_values(values, prefix):
    """Return mean, SD, SE and a normal-approximation 95% CI."""
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)

    if n == 0:
        return {
            f"{prefix} Mean": np.nan,
            f"{prefix} SD": np.nan,
            f"{prefix} SE": np.nan,
            f"{prefix} CI95 Lower": np.nan,
            f"{prefix} CI95 Upper": np.nan
        }

    mean_val = float(np.mean(arr))
    if n > 1:
        sd_val = float(np.std(arr, ddof=1))
        se_val = sd_val / math.sqrt(n)
        ci_low = mean_val - 1.96 * se_val
        ci_high = mean_val + 1.96 * se_val
    else:
        sd_val = np.nan
        se_val = np.nan
        ci_low = np.nan
        ci_high = np.nan

    return {
        f"{prefix} Mean": mean_val,
        f"{prefix} SD": sd_val,
        f"{prefix} SE": se_val,
        f"{prefix} CI95 Lower": ci_low,
        f"{prefix} CI95 Upper": ci_high
    }


def _summarize_vaccinated_population(vax_mask):
    """Describe who is reached by an allocation mask without changing allocation."""
    vaccinated_indices = np.where(vax_mask)[0]
    vaccinated_count = int(len(vaccinated_indices))

    if vaccinated_count == 0:
        mean_age = np.nan
        mean_strength = np.nan
        mean_neighbour_strength = np.nan
        mean_mobility = np.nan
        key_group_share = np.nan
    else:
        mean_age = float(np.mean(df['age_num'].values[vaccinated_indices]))
        mean_strength = float(np.mean(total_node_strength[vaccinated_indices]))
        mean_neighbour_strength = float(np.mean(avg_neighbor_strength[vaccinated_indices]))
        mean_mobility = float(np.mean(mobility_index[vaccinated_indices]))
        key_group_share = float(np.mean(is_key_group[vaccinated_indices]))

    households_reached = 0
    fully_vaccinated_households = 0
    partially_vaccinated_households = 0

    for members in house2members.values():
        member_idx = np.asarray(members, dtype=int)
        vaccinated_in_household = int(np.sum(vax_mask[member_idx]))

        if vaccinated_in_household > 0:
            households_reached += 1
            if vaccinated_in_household == len(member_idx):
                fully_vaccinated_households += 1
            else:
                partially_vaccinated_households += 1

    return {
        "Vaccinated Mean Age": mean_age,
        "Vaccinated Mean Total Strength": mean_strength,
        "Vaccinated Mean Neighbour Strength": mean_neighbour_strength,
        "Vaccinated Mean Mobility": mean_mobility,
        "Vaccinated Key-group Share": key_group_share,
        "Households Reached": households_reached,
        "Fully Vaccinated Households": fully_vaccinated_households,
        "Partially Vaccinated Households": partially_vaccinated_households
    }


def _describe_feature_group(values_dict, status, current_ves0, target_r0,
                            coverage, actual_coverage, strategy):
    """
    Retain exact descriptive summaries of the pooled node-run observations.
    These summaries are descriptive only because the same node can appear in
    multiple Monte Carlo repetitions.
    """
    output = {
        "VES_0": current_ves0,
        "R0": target_r0,
        "Coverage": coverage,
        "Actual Coverage": actual_coverage,
        "Strategy": strategy,
        "Status": status
    }

    first_key = next(iter(values_dict))
    output["Pooled Node-run Count"] = int(len(values_dict[first_key]))

    for metric_name, values in values_dict.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            output.update({
                f"{metric_name} Mean": np.nan,
                f"{metric_name} SD": np.nan,
                f"{metric_name} Q25": np.nan,
                f"{metric_name} Median": np.nan,
                f"{metric_name} Q75": np.nan
            })
        else:
            output.update({
                f"{metric_name} Mean": float(np.mean(arr)),
                f"{metric_name} SD": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
                f"{metric_name} Q25": float(np.quantile(arr, 0.25)),
                f"{metric_name} Median": float(np.quantile(arr, 0.50)),
                f"{metric_name} Q75": float(np.quantile(arr, 0.75))
            })

    return output


# ============================================
# Parallel worker: one VES_0 x R0 scenario
# ============================================
def run_single_scenario(current_ves0, target_r0):
    # Retain the original seed-reset behaviour used by the supplied simulation code.
    np.random.seed(RNG_SEED)
    random.seed(RNG_SEED)

    print(f"[WORKER START] VES_0={current_ves0}, R0={target_r0}", flush=True)
    start_time = time.time()

    # 1. Mean-field compensation used in the original calibration
    sum_internal = np.sum(strength_internal)
    work_exposure = np.sum(strength_vir_work[is_key_group]) * EXTERNAL_PREVALENCE
    social_exposure = np.sum(strength_vir_social[daily_hours_nonfixed > 0]) * EXTERNAL_PREVALENCE
    sum_external = work_exposure + social_exposure

    internal_ratio = sum_internal / (sum_internal + sum_external + 1e-9)
    adjusted_target_r0 = target_r0 * internal_ratio
    current_beta = calibrate_beta(adjusted_target_r0, PROB_I_TO_S, G)

    # 2. Scenario-level storage
    temp_r0_results = []
    feature_dist_data = []
    feature_status_summary_data = []
    feature_bin_data = []
    uncertainty_replicate_data = []

    # 3. Run all coverage levels and strategies
    for cov_idx, cov in enumerate(COVERAGE_STEPS, start=1):
        print(
            f"[PROGRESS] VES_0={current_ves0}, R0={target_r0}, "
            f"Coverage {cov_idx}/{len(COVERAGE_STEPS)} = {cov:.2f}, "
            f"Elapsed = {(time.time() - start_time) / 60:.1f} min",
            flush=True
        )

        is_detailed_feature_coverage = any(
            np.isclose(cov, target_cov, atol=0.01)
            for target_cov in DETAILED_FEATURE_COVERAGES
        )
        uncertainty_setting = _match_selected_uncertainty_setting(
            current_ves0, target_r0, cov
        )

        for strat in strategies_list:
            vax_mask = get_vaccinated_mask(strat, cov, N)
            actual_num_vax = int(np.sum(vax_mask))
            actual_coverage = actual_num_vax / N

            t_rate, t_peak_sz, t_peak_tm, t_death = [], [], [], []

            # Retain the original pooled feature samples at 20%, 50% and 80%.
            strat_inf_str, strat_uninf_str = [], []
            strat_inf_mob, strat_uninf_mob = [], []
            strat_inf_age, strat_uninf_age = [], []
            strat_inf_nbr, strat_uninf_nbr = [], []
            strat_inf_key, strat_uninf_key = [], []
            strat_inf_vax, strat_uninf_vax = [], []
            strat_inf_exp, strat_uninf_exp = [], []

            retain_exact_bins_for_strategy = (
                strat in FIG4_BIN_STRATEGIES
                or is_detailed_feature_coverage
            )

            for mc_run_id in range(1, MC_RUNS_PER_POINT + 1):
                (
                    c,
                    p_sz,
                    p_tm,
                    d,
                    daily_curve,
                    ever_infected_mask,
                    cumulative_exposure_arr
                ) = run_strategy_simulation(current_beta, vax_mask, current_ves0)

                infection_rate = c / N
                t_rate.append(infection_rate)
                t_peak_sz.append(p_sz)
                t_peak_tm.append(p_tm)
                t_death.append(d)

                # Replicate-level outcomes for pre-specified uncertainty settings only.
                # At target coverage, retain the three principal strategies.
                # At coverage=0, retain Random only as the common baseline Y0.
                retain_uncertainty_outcome = False
                if uncertainty_setting is not None:
                    uncertainty_scenario_label, uncertainty_data_role = uncertainty_setting
                    retain_uncertainty_outcome = (
                        (uncertainty_data_role == "Target coverage" and strat in UNCERTAINTY_STRATEGIES)
                        or
                        (uncertainty_data_role == "No-vaccination baseline" and strat == "Random")
                    )

                if retain_uncertainty_outcome:
                    uncertainty_replicate_data.append({
                        "Scenario Label": uncertainty_scenario_label,
                        "Data Role": uncertainty_data_role,
                        "VES_0": current_ves0,
                        "R0": target_r0,
                        "Coverage": cov,
                        "Actual Coverage": actual_coverage,
                        "Strategy": strat,
                        "MC_Run": mc_run_id,
                        "Infection Rate": infection_rate,
                        "Peak Size": p_sz,
                        "Peak Time": p_tm,
                        "Expected Deaths": d
                    })

                # Exact Fig. 4-style bin counts. These are retained for Random and
                # High Strength (Node) at every coverage and for every strategy at
                # the three detailed coverages.
                if retain_exact_bins_for_strategy:
                    for bin_definition in feature_bin_definitions:
                        internal_bin_id = bin_definition["Internal_Bin_ID"]
                        node_indices_bin = feature_bin_node_indices[internal_bin_id]
                        total_nodes_bin = int(len(node_indices_bin))

                        if total_nodes_bin == 0:
                            infected_nodes_bin = 0
                            uninfected_nodes_bin = 0
                            uninfected_proportion_bin = np.nan
                        else:
                            infected_nodes_bin = int(
                                np.sum(ever_infected_mask[node_indices_bin])
                            )
                            uninfected_nodes_bin = (
                                total_nodes_bin - infected_nodes_bin
                            )
                            uninfected_proportion_bin = (
                                uninfected_nodes_bin / total_nodes_bin
                            )

                        feature_bin_data.append({
                            "VES_0": current_ves0,
                            "R0": target_r0,
                            "Coverage": cov,
                            "Actual Coverage": actual_coverage,
                            "Strategy": strat,
                            "MC_Run": mc_run_id,
                            "Bin_ID": bin_definition["Bin_ID"],
                            "Bin_Type": bin_definition["Bin_Type"],
                            "Bin_Left": bin_definition["Bin_Left"],
                            "Bin_Right": bin_definition["Bin_Right"],
                            "Bin_Center": bin_definition["Bin_Center"],
                            "Total_Nodes": total_nodes_bin,
                            "Infected_Nodes": infected_nodes_bin,
                            "Uninfected_Nodes": uninfected_nodes_bin,
                            "Uninfected_Proportion": uninfected_proportion_bin,
                            "Node_Weight": bin_definition["Node_Weight"]
                        })

                # Pooled feature values at 20%, 50% and 80% coverage.
                if is_detailed_feature_coverage:
                    inf_idx = np.where(ever_infected_mask)[0]
                    uninf_idx = np.where(~ever_infected_mask)[0]

                    strat_inf_str.extend(total_node_strength[inf_idx])
                    strat_uninf_str.extend(total_node_strength[uninf_idx])
                    strat_inf_mob.extend(mobility_index[inf_idx])
                    strat_uninf_mob.extend(mobility_index[uninf_idx])
                    strat_inf_age.extend(df['age_num'].values[inf_idx])
                    strat_uninf_age.extend(df['age_num'].values[uninf_idx])
                    strat_inf_nbr.extend(avg_neighbor_strength[inf_idx])
                    strat_uninf_nbr.extend(avg_neighbor_strength[uninf_idx])
                    strat_inf_key.extend(is_key_group[inf_idx].astype(int))
                    strat_uninf_key.extend(is_key_group[uninf_idx].astype(int))
                    strat_inf_vax.extend(vax_mask[inf_idx].astype(int))
                    strat_uninf_vax.extend(vax_mask[uninf_idx].astype(int))
                    strat_inf_exp.extend(cumulative_exposure_arr[inf_idx])
                    strat_uninf_exp.extend(cumulative_exposure_arr[uninf_idx])

            # Scenario mean plus across-repetition uncertainty summaries
            record = {
                "VES_0": current_ves0,
                "R0": target_r0,
                "Coverage": cov,
                "Actual Coverage": actual_coverage,
                "Actual Vaccinated Count": actual_num_vax,
                "Strategy": strat,
                "Calibrated Beta": current_beta,
                "Internal Exposure Ratio": internal_ratio,
                "Is Common Baseline Source": (
                    strat == "Random" and np.isclose(cov, 0.0, atol=1e-12)
                )
            }
            # Preserve the original mean-column names used by the existing
            # plotting scripts, while adding SD, SE and 95% CI columns.
            record.update({
                "Infection Rate": float(np.mean(t_rate)),
                "Peak Size": float(np.mean(t_peak_sz)),
                "Peak Time": float(np.mean(t_peak_tm)),
                "Expected Deaths": float(np.mean(t_death))
            })
            infection_summary = _summarize_values(t_rate, "Infection Rate")
            peak_size_summary = _summarize_values(t_peak_sz, "Peak Size")
            peak_time_summary = _summarize_values(t_peak_tm, "Peak Time")
            death_summary = _summarize_values(t_death, "Expected Deaths")

            # The mean aliases are omitted because the original mean columns above
            # are retained for backward compatibility.
            infection_summary.pop("Infection Rate Mean", None)
            peak_size_summary.pop("Peak Size Mean", None)
            peak_time_summary.pop("Peak Time Mean", None)
            death_summary.pop("Expected Deaths Mean", None)

            record.update(infection_summary)
            record.update(peak_size_summary)
            record.update(peak_time_summary)
            record.update(death_summary)
            record.update(_summarize_vaccinated_population(vax_mask))
            temp_r0_results.append(record)

            # Exact pooled descriptive summaries are retained before sampling.
            if is_detailed_feature_coverage:
                infected_values = {
                    "Total Strength": strat_inf_str,
                    "Average Neighbour Strength": strat_inf_nbr,
                    "Mobility": strat_inf_mob,
                    "Age": strat_inf_age,
                    "Key-group Indicator": strat_inf_key,
                    "Vaccinated Indicator": strat_inf_vax,
                    "Cumulative Exposure": strat_inf_exp
                }
                uninfected_values = {
                    "Total Strength": strat_uninf_str,
                    "Average Neighbour Strength": strat_uninf_nbr,
                    "Mobility": strat_uninf_mob,
                    "Age": strat_uninf_age,
                    "Key-group Indicator": strat_uninf_key,
                    "Vaccinated Indicator": strat_uninf_vax,
                    "Cumulative Exposure": strat_uninf_exp
                }

                feature_status_summary_data.append(
                    _describe_feature_group(
                        infected_values,
                        "Infected",
                        current_ves0,
                        target_r0,
                        cov,
                        actual_coverage,
                        strat
                    )
                )
                feature_status_summary_data.append(
                    _describe_feature_group(
                        uninfected_values,
                        "Uninfected",
                        current_ves0,
                        target_r0,
                        cov,
                        actual_coverage,
                        strat
                    )
                )

                # Keep the original stratified visualization sample. Sampling is
                # performed after all 100 repetitions for the current strategy,
                # preserving the supplied script's ordering of random-number use.
                if SAVE_SAMPLED_NODE_DISTRIBUTIONS:
                    df_inf = pd.DataFrame({
                        'Coverage': cov,
                        'Strength': strat_inf_str,
                        'Mobility': strat_inf_mob,
                        'Age': strat_inf_age,
                        'Avg_Neighbor_Strength': strat_inf_nbr,
                        'Is_Key_Group': strat_inf_key,
                        'Is_Vaccinated': strat_inf_vax,
                        'Cumulative_Exposure': strat_inf_exp,
                        'Status': 'Infected',
                        'Strategy': strat
                    })
                    df_uninf = pd.DataFrame({
                        'Coverage': cov,
                        'Strength': strat_uninf_str,
                        'Mobility': strat_uninf_mob,
                        'Age': strat_uninf_age,
                        'Avg_Neighbor_Strength': strat_uninf_nbr,
                        'Is_Key_Group': strat_uninf_key,
                        'Is_Vaccinated': strat_uninf_vax,
                        'Cumulative_Exposure': strat_uninf_exp,
                        'Status': 'Uninfected',
                        'Strategy': strat
                    })

                    if len(df_inf) > SAMPLED_N_PER_STATUS:
                        df_inf = df_inf.sample(SAMPLED_N_PER_STATUS)
                    if len(df_uninf) > SAMPLED_N_PER_STATUS:
                        df_uninf = df_uninf.sample(SAMPLED_N_PER_STATUS)

                    feature_dist_data.append(pd.concat([df_inf, df_uninf]))

    # 4. Save scenario-level files. CSV/GZIP is used for large retained tables
    # because Excel output is substantially slower and has a row limit.
    scenario_tag = f"VES{current_ves0}_R0_{target_r0}"

    df_current_r0 = pd.DataFrame(temp_r0_results)
    df_current_r0.to_csv(
        os.path.join(
            SCENARIO_SUMMARY_DIR,
            f"Scenario_Summary_{scenario_tag}.csv"
        ),
        index=False
    )

    if feature_bin_data:
        pd.DataFrame(feature_bin_data).to_csv(
            os.path.join(
                FIG4_BIN_DIR,
                f"Feature_Bin_Exact_AllCoverages_{scenario_tag}.csv.gz"
            ),
            index=False,
            compression="gzip"
        )

    if feature_status_summary_data:
        pd.DataFrame(feature_status_summary_data).to_csv(
            os.path.join(
                FEATURE_SUMMARY_DIR,
                f"Feature_Status_Exact_Summary_{scenario_tag}.csv.gz"
            ),
            index=False,
            compression="gzip"
        )

    if feature_dist_data:
        pd.concat(feature_dist_data, ignore_index=True).to_csv(
            os.path.join(
                SAMPLED_FEATURE_DIR,
                f"Feature_Distribution_Sampled_Cov20_50_80_{scenario_tag}.csv.gz"
            ),
            index=False,
            compression="gzip"
        )

    if uncertainty_replicate_data:
        pd.DataFrame(uncertainty_replicate_data).to_csv(
            os.path.join(
                UNCERTAINTY_REPLICATE_DIR,
                f"Selected_Replicate_Outcomes_{scenario_tag}.csv.gz"
            ),
            index=False,
            compression="gzip"
        )

    print(
        f"[WORKER DONE] VES_0={current_ves0}, R0={target_r0} Complete. "
        f"Elapsed = {(time.time() - start_time) / 60:.1f} min",
        flush=True
    )
    return df_current_r0


# ============================================
# Parallel scheduling and final export
# ============================================
if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("INITIATING PARALLEL EXECUTION")
    print(f"{'='*60}")

    # Save network-level metadata once. Raw addresses are not exported.
    household_codes, _ = pd.factorize(df['_household_key'])
    node_attribute_df = pd.DataFrame({
        "Node_ID": np.arange(N, dtype=int),
        "Anonymous_Household_ID": household_codes.astype(int),
        "Age": df['age_num'].values,
        "Work_School_Activity_Status": is_key_group.astype(int),
        "Daily_Activity_Duration": mobility_index,
        "Internal_Weighted_Strength": strength_internal,
        "External_Virtual_Strength": strength_virtual,
        "Total_Weighted_Exposure_Strength": total_node_strength,
        "Average_Neighbour_Exposure_Strength": avg_neighbor_strength
    })
    node_attribute_df.to_csv(
        os.path.join(METADATA_DIR, "Network_Node_Attributes.csv.gz"),
        index=False,
        compression="gzip"
    )

    pd.DataFrame(feature_bin_definitions).drop(
        columns=["Internal_Bin_ID"]
    ).to_csv(
        os.path.join(METADATA_DIR, "Exposure_Strength_Bin_Definitions.csv"),
        index=False
    )

    configuration_rows = [
        ("N", N),
        ("RNG_SEED", RNG_SEED),
        ("MC_RUNS_PER_POINT", MC_RUNS_PER_POINT),
        ("SIM_DAYS_STRATEGY", SIM_DAYS_STRATEGY),
        ("NUM_SEEDS_PER_RUN", NUM_SEEDS_PER_RUN),
        ("PROB_E_TO_I", PROB_E_TO_I),
        ("PROB_I_TO_S", PROB_I_TO_S),
        ("NAT_EFF_MAX", NAT_EFF_MAX),
        ("NAT_EFF_MIN", NAT_EFF_MIN),
        ("EXTERNAL_PREVALENCE", EXTERNAL_PREVALENCE),
        ("FEATURE_BIN_COUNT", FEATURE_BIN_COUNT),
        ("VES_0_LIST", repr(VES_0_LIST)),
        ("R0_LIST", repr(R0_LIST)),
        ("COVERAGE_STEPS", repr(COVERAGE_STEPS.tolist())),
        ("STRATEGIES", repr(strategies_list)),
        ("UNCERTAINTY_SELECTED_SCENARIOS", repr(UNCERTAINTY_SELECTED_SCENARIOS))
    ]
    pd.DataFrame(
        configuration_rows,
        columns=["Parameter", "Value"]
    ).to_csv(
        os.path.join(METADATA_DIR, "Run_Configuration.csv"),
        index=False
    )

    tasks = list(itertools.product(VES_0_LIST, R0_LIST))
    print(f"[INFO] Total scenarios to compute: {len(tasks)}")

    all_results_df_list = []

    if TEST_MODE:
        for ves0, r0 in tasks:
            all_results_df_list.append(run_single_scenario(ves0, r0))
    else:
        with ProcessPoolExecutor(max_workers=MAX_CORES) as executor:
            futures = {
                executor.submit(run_single_scenario, ves0, r0): (ves0, r0)
                for ves0, r0 in tasks
            }

            for future in as_completed(futures):
                ves0, r0 = futures[future]
                try:
                    result_df = future.result()
                    all_results_df_list.append(result_df)
                except Exception as exc:
                    print(
                        f"[ERROR] Task VES0={ves0}, R0={r0} "
                        f"generated an exception: {exc}",
                        flush=True
                    )

    if all_results_df_list:
        final_all_results_df = pd.concat(
            all_results_df_list,
            ignore_index=True
        ).sort_values(
            ["VES_0", "R0", "Coverage", "Strategy"]
        )

        expected_summary_rows = (
            len(VES_0_LIST)
            * len(R0_LIST)
            * len(COVERAGE_STEPS)
            * len(strategies_list)
        )
        if len(final_all_results_df) != expected_summary_rows:
            raise RuntimeError(
                "Incomplete primary-simulation summary: expected "
                f"{expected_summary_rows} rows but found "
                f"{len(final_all_results_df)}."
            )

        final_all_results_df.to_csv(
            os.path.join(
                OUTPUT_ROOT,
                "Simulation_Results_All_Scenarios_With_Uncertainty.csv.gz"
            ),
            index=False,
            compression="gzip"
        )

        # Preserve the original Excel filename expected by the existing
        # downstream plotting scripts.
        final_all_results_df.to_excel(
            os.path.join(
                OUTPUT_ROOT,
                "Simulation_Results_All_Scenarios_Parallel.xlsx"
            ),
            index=False
        )

        print(
            "[SAVED] Combined scenario summaries saved in CSV.GZ and Excel formats.",
            flush=True
        )

    if TEST_MODE:
        print("\n[TEST MODE PASSED] Synthetic smoke-test outputs were generated successfully.")
    else:
        print("\n[DONE] Parallel computing and data retention finished!")
