"""Run robustness experiments across alternative disease-course and immunity parameterizations.

The script reuses the survey-derived exposure network and evaluates the random,
high-exposure, and oldest-first vaccination strategies under COVID-like,
influenza-like, and RSV-like parameter scenarios.
"""

import os
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
import hashlib
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
DATA_FILE = '20231012.xlsx'
DATA_PATH = str(PROJECT_ROOT / 'data' / DATA_FILE)
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'disease_parameter_robustness'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHEET_ID = 0

# Parallel execution settings
MAX_CORES = int(os.environ.get('SLURM_CPUS_PER_TASK', 20))

# ---- Disease-like robustness scenarios ----
# The scenario dictionary groups disease-course and immune-waning assumptions.
# These are literature-informed robustness parameterizations, not pathogen-specific calibrated forecasts.
# Mean E duration ~= 1 / PROB_E_TO_I; mean I duration ~= 1 / PROB_I_TO_S.
SCENARIOS = {
    "COVID_like": {
        "PROB_E_TO_I": 0.20,
        "PROB_I_TO_S": 0.142,
        "VES_0_LIST": [0.20, 0.50, 0.80],
        "R0_LIST": [1.0, 2.0, 4.0, 8.0],
        "VAX_FULL_DAYS": 100,
        "VAX_MID_DAYS": 180,
        "VAX_MID_MULT": 0.70,
        "VAX_LONG_MULT": 0.40,
        "NAT_EFF_MAX": 0.652,
        "NAT_EFF_MIN": 0.247,
        "NAT_FULL_DAYS": 90,
        "NAT_WANE_DAYS": 365,
    },
    "Influenza_like": {
        "PROB_E_TO_I": 0.50,
        "PROB_I_TO_S": 0.20,
        "VES_0_LIST": [0.20, 0.50, 0.80],
        "R0_LIST": [1.0, 2.0, 4.0, 8.0],
        "VAX_FULL_DAYS": 60,
        "VAX_MID_DAYS": 120,
        "VAX_MID_MULT": 0.70,
        "VAX_LONG_MULT": 0.40,
        "NAT_EFF_MAX": 0.50,
        "NAT_EFF_MIN": 0.10,
        "NAT_FULL_DAYS": 60,
        "NAT_WANE_DAYS": 180,
    },
    "RSV_like": {
        "PROB_E_TO_I": 0.25,
        "PROB_I_TO_S": 0.167,
        "VES_0_LIST": [0.20, 0.50, 0.80],
        "R0_LIST": [1.0, 2.0, 4.0, 8.0],
        "VAX_FULL_DAYS": 90,
        "VAX_MID_DAYS": 180,
        "VAX_MID_MULT": 0.70,
        "VAX_LONG_MULT": 0.40,
        "NAT_EFF_MAX": 0.40,
        "NAT_EFF_MIN": 0.05,
        "NAT_FULL_DAYS": 60,
        "NAT_WANE_DAYS": 180,
    },
}

# ---- Network & Time Parameters ----
HH_CONTACT_HOURS = 12.0 
SOCIAL_ENCOUNTER_DURATION = 0.25 
INTERNAL_SOCIAL_RATIO = 0.3

# ---- External Virtual Parameters ----
VIRTUAL_CONTACT_MAPPING = {'a': 3.0, 'b': 10.0, 'c': 20.0, 'd': 30.0}
EXTERNAL_PREVALENCE = 0.005
EXT_SOCIAL_CONTACT_DENSITY = 4.0 
NUM_SEEDS_PER_RUN = 10
RNG_SEED = 42

MC_RUNS_PER_POINT = 100
SIM_DAYS_STRATEGY = 500

# Data-retention settings. Dynamics curves are not required for the planned SI figures.
SAVE_DYNAMICS_CURVES = False
SAVE_REPLICATE_OUTCOMES = True
SAVE_AGE_GROUP_COUNTS = True

# For robustness analysis we only need no-vaccination and 50% coverage.
# Add 0.2 and 0.8 here if you also want coverage robustness in the same run.
COVERAGE_STEPS = np.array([0.0, 0.50])

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


def make_stable_seed(*items, base_seed=RNG_SEED):
    """Create a deterministic seed that is stable across Python sessions."""
    key = "|".join(map(str, items))
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int((base_seed + int(digest[:8], 16)) % (2**32 - 1))


def summarize_metric(values, prefix):
    """Return mean, SD, SE and a normal-approximation 95% CI."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean_val = float(np.mean(arr)) if n else np.nan
    sd_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se_val = sd_val / np.sqrt(n) if n > 0 else np.nan
    return {
        prefix: mean_val,
        f"{prefix} SD": sd_val,
        f"{prefix} SE": se_val,
        f"{prefix} CI95 Lower": mean_val - 1.96 * se_val if n > 0 else np.nan,
        f"{prefix} CI95 Upper": mean_val + 1.96 * se_val if n > 0 else np.nan,
    }


def get_age_group_infection_counts(ever_infected_mask, vax_mask):
    """Retain unique ever-infected counts by age group and vaccination status."""
    counts = {}
    for age in AGE_GROUP_VALUES:
        label = AGE_GROUP_LABELS[age]
        age_mask = np.isclose(AGE_NUM_VALUES, age)
        infected_age = ever_infected_mask & age_mask
        counts[f"Infected_Age_{label}"] = int(np.sum(infected_age))
        counts[f"Vaccinated_Infected_Age_{label}"] = int(np.sum(infected_age & vax_mask))
        counts[f"Unvaccinated_Infected_Age_{label}"] = int(np.sum(infected_age & (~vax_mask)))
    return counts


# =============== Data Loading ===============
if not os.path.exists(DATA_PATH): raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_ID)
N = len(df)
print(f"[INFO] Loaded N = {N}")

COL_COMMUNITY = pick_column(df, ["小区名称", "小区", "社区名称", "社区"])
COL_ADDRESS = pick_column(df, ["住户详细地址", "住址"])
COL_WORK_STATUS = pick_column(df, ["工作状态"])
COL_CONTACT_COUNT = pick_column(df, ["人员数量", "学习场所的人员数量"])
COL_TRIP_FREQ = [f"频率{i}" for i in range(1, 6)]
COL_TRIP_DUR = [f"时长{i}" for i in range(1, 6)]
COL_AGE = pick_column(df, ["年龄"])

# Household definition used for network construction:
# members sharing the cleaned community name and detailed address are treated as co-residents.
df['_community_clean'] = df[COL_COMMUNITY].astype(str).str.strip()
df['_address_clean'] = df[COL_ADDRESS].astype(str).str.strip()
df['_household_key'] = df['_community_clean'] + '|' + df['_address_clean']

age_map_dict = {'a':15, 'b':24, 'c':34.5, 'd':44.5, 'e':54.5, 'f':64.5, 'g':75}
df['age_num'] = df[COL_AGE].astype(str).str.strip().map(age_map_dict).fillna(45.0)

# Age-group counts are retained for post-processing with alternative disease-specific
# fatality-risk functions. The in-model expected-deaths outcome continues to use IFR_MAP
# for comparability with the main COVID-like baseline analysis.
AGE_NUM_VALUES = df['age_num'].values.astype(float)
AGE_GROUP_VALUES = np.array(sorted(np.unique(AGE_NUM_VALUES)), dtype=float)
AGE_GROUP_LABELS = {
    age: str(age).replace('.', '_') for age in AGE_GROUP_VALUES
}
AGE_GROUP_POPULATION = {
    f"Population_Age_{AGE_GROUP_LABELS[age]}": int(np.sum(np.isclose(AGE_NUM_VALUES, age)))
    for age in AGE_GROUP_VALUES
}

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

def run_strategy_simulation(beta_val, vax_mask, ves_0, scenario_params):
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
        if d <= scenario_params["VAX_FULL_DAYS"]:
            current_ve = ves_0
        elif d <= scenario_params["VAX_MID_DAYS"]:
            current_ve = scenario_params["VAX_MID_MULT"] * ves_0
        else:
            current_ve = scenario_params["VAX_LONG_MULT"] * ves_0

        vax_susc_mult = np.where(vax_mask, 1.0 - current_ve, 1.0)
        
        # ==========================================
        # ==========================================
        nat_susc_mult = np.ones(N)

        nat_full_days = scenario_params["NAT_FULL_DAYS"]
        nat_wane_days = scenario_params["NAT_WANE_DAYS"]
        nat_eff_max = scenario_params["NAT_EFF_MAX"]
        nat_eff_min = scenario_params["NAT_EFF_MIN"]

        mask_full = (days_since_rec >= 0) & (days_since_rec <= nat_full_days)
        mask_wane = (days_since_rec > nat_full_days) & (days_since_rec <= nat_wane_days)
        mask_long = (days_since_rec > nat_wane_days)

        nat_susc_mult[mask_full] = 1.0 - nat_eff_max

        if np.any(mask_wane):
            decay_ratio = (days_since_rec[mask_wane] - nat_full_days) / max(nat_wane_days - nat_full_days, 1)
            current_efficacy = nat_eff_max - (nat_eff_max - nat_eff_min) * decay_ratio
            nat_susc_mult[mask_wane] = 1.0 - current_efficacy

        nat_susc_mult[mask_long] = 1.0 - nat_eff_min
        
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
        new_state[exp_indices] = np.where(r_val[exp_indices] < scenario_params["PROB_E_TO_I"], 2, 1)
        
        inf_indices = np.where(state == 2)[0]
        recovered = r_rec[inf_indices] < scenario_params["PROB_I_TO_S"]
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
    peak_tm = (np.argmax(daily_infected_counts) + 1) if len(daily_infected_counts) > 0 else 0
    infected_indices = np.where(ever_infected)[0]
    expected_deaths = calculate_expected_deaths(infected_indices, df['age_num'])
    
    return unique_infected_total, peak_sz, peak_tm, expected_deaths, daily_infected_counts, ever_infected, cumulative_exposure

# ============================================
# MAIN EXECUTION LOOP FOR DISEASE-LIKE ROBUSTNESS
# ============================================

# For disease-like robustness, keep only the strategies needed to defend the main conclusion.
# Additional strategies can remain in get_vaccinated_mask(), but are not run here.
strategies_list = [
    'Random',
    'High Strength (Node)',
    'Priority: Oldest First',
]

plot_strategies = strategies_list.copy()
strat_colors = dict(zip(plot_strategies, sns.color_palette("deep", n_colors=len(plot_strategies))))

OUTCOME_COLUMNS = [
    "Final Infection Proportion",
    "Epidemic Peak Size",
    "Expected Deaths",
]

print(f"\n[INFO] Starting disease-like robustness simulation")
print(f"[INFO] Scenarios: {list(SCENARIOS.keys())}")


def run_single_scenario(scenario_name, current_ves0, target_r0):
    """Run one disease-like parameter scenario for a given VES_0 and R0."""
    scenario_params = SCENARIOS[scenario_name]

    # Deterministic worker-level seeding for reproducibility.
    # Strategy simulations remain independent rather than paired.
    stable_seed = make_stable_seed(scenario_name, current_ves0, target_r0)
    np.random.seed(stable_seed)
    random.seed(stable_seed)

    print(
        f"[WORKER START] Scenario={scenario_name}, VES_0={current_ves0}, R0={target_r0}",
        flush=True
    )
    start_time = time.time()

    # Mean-field compensation for external exposure, unchanged across disease-like scenarios.
    sum_internal = np.sum(strength_internal)
    work_exposure = np.sum(strength_vir_work[is_key_group]) * EXTERNAL_PREVALENCE
    social_exposure = np.sum(strength_vir_social[daily_hours_nonfixed > 0]) * EXTERNAL_PREVALENCE
    sum_external = work_exposure + social_exposure

    internal_ratio = sum_internal / (sum_internal + sum_external + 1e-9)
    adjusted_target_r0 = target_r0 * internal_ratio

    # Beta calibration uses the scenario-specific recovery probability.
    current_beta = calibrate_beta(
        adjusted_target_r0,
        scenario_params["PROB_I_TO_S"],
        G
    )

    temp_results = []
    replicate_records = []
    dynamics_records = []
    temporal_curves_storage = defaultdict(list)
    target_plot_coverage = 0.50

    for cov_idx, cov in enumerate(COVERAGE_STEPS, start=1):
        print(
            f"[PROGRESS] Scenario={scenario_name}, VES_0={current_ves0}, R0={target_r0}, "
            f"Coverage {cov_idx}/{len(COVERAGE_STEPS)} = {cov:.2f}, "
            f"Elapsed = {(time.time() - start_time) / 60:.1f} min",
            flush=True
        )

        is_target_point = np.isclose(cov, target_plot_coverage, atol=0.01)
        strategies_to_run = ['Random'] if np.isclose(cov, 0.0, atol=0.01) else strategies_list

        for strat in strategies_to_run:
            # Vaccinated set is generated once per setting and held fixed across MC repetitions.
            vax_mask = get_vaccinated_mask(strat, cov, N)
            actual_num_vax = int(np.sum(vax_mask))
            actual_coverage = actual_num_vax / N

            final_inf_prop, peak_sz, peak_tm, exp_deaths = [], [], [], []

            for mc_run in range(1, MC_RUNS_PER_POINT + 1):
                c, p_sz, p_tm, d, daily_curve, ever_infected_mask, _ = run_strategy_simulation(
                    current_beta,
                    vax_mask,
                    current_ves0,
                    scenario_params
                )
                final_value = c / N
                final_inf_prop.append(final_value)
                peak_sz.append(p_sz)
                peak_tm.append(p_tm)
                exp_deaths.append(d)

                if SAVE_DYNAMICS_CURVES and is_target_point:
                    temporal_curves_storage[strat].append(daily_curve)

                if SAVE_REPLICATE_OUTCOMES:
                    replicate_row = {
                        "Scenario": scenario_name,
                        "VES_0": current_ves0,
                        "R0": target_r0,
                        "Coverage": float(cov),
                        "Actual Coverage": actual_coverage,
                        "Actual Vaccinated Count": actual_num_vax,
                        "Strategy": strat,
                        "MC_Run": mc_run,
                        "Final Infection Proportion": float(final_value),
                        "Epidemic Peak Size": float(p_sz),
                        "Peak Time": float(p_tm),
                        "Expected Deaths": float(d),
                        "IFR_Structure": "COVID-like baseline age-specific IFR",
                    }
                    if SAVE_AGE_GROUP_COUNTS:
                        replicate_row.update(get_age_group_infection_counts(ever_infected_mask, vax_mask))
                    replicate_records.append(replicate_row)

            record = {
                "Scenario": scenario_name,
                "VES_0": current_ves0,
                "R0": target_r0,
                "Coverage": float(cov),
                "Actual Coverage": actual_coverage,
                "Actual Vaccinated Count": actual_num_vax,
                "Strategy": strat,
                **summarize_metric(final_inf_prop, "Final Infection Proportion"),
                **summarize_metric(peak_sz, "Epidemic Peak Size"),
                **summarize_metric(peak_tm, "Peak Time"),
                **summarize_metric(exp_deaths, "Expected Deaths"),
                "PROB_E_TO_I": scenario_params["PROB_E_TO_I"],
                "PROB_I_TO_S": scenario_params["PROB_I_TO_S"],
                "VAX_FULL_DAYS": scenario_params["VAX_FULL_DAYS"],
                "VAX_MID_DAYS": scenario_params["VAX_MID_DAYS"],
                "VAX_MID_MULT": scenario_params["VAX_MID_MULT"],
                "VAX_LONG_MULT": scenario_params["VAX_LONG_MULT"],
                "NAT_EFF_MAX": scenario_params["NAT_EFF_MAX"],
                "NAT_EFF_MIN": scenario_params["NAT_EFF_MIN"],
                "NAT_FULL_DAYS": scenario_params["NAT_FULL_DAYS"],
                "NAT_WANE_DAYS": scenario_params["NAT_WANE_DAYS"],
                "External Prevalence": EXTERNAL_PREVALENCE,
                "IFR_Structure": "COVID-like baseline age-specific IFR",
                "MC Runs": MC_RUNS_PER_POINT,
                "Simulation Days": SIM_DAYS_STRATEGY,
                "Initial Infectious Nodes": NUM_SEEDS_PER_RUN,
            }
            temp_results.append(record)

    if SAVE_DYNAMICS_CURVES and temporal_curves_storage:
        for strat in plot_strategies:
            curves = temporal_curves_storage.get(strat, [])
            if len(curves) == 0:
                continue
            curves_matrix = np.vstack(curves)
            mean_curve = np.mean(curves_matrix, axis=0)
            std_curve = np.std(curves_matrix, axis=0)
            for day, mean_i, sd_i in zip(np.arange(1, len(mean_curve) + 1), mean_curve, std_curve):
                dynamics_records.append({
                    "Scenario": scenario_name,
                    "VES_0": current_ves0,
                    "R0": target_r0,
                    "Coverage": target_plot_coverage,
                    "Strategy": strat,
                    "Day": int(day),
                    "Mean Infectious": float(mean_i),
                    "SD Infectious": float(sd_i),
                })

    print(f"[WORKER DONE] Scenario={scenario_name}, VES_0={current_ves0}, R0={target_r0}")
    return (
        pd.DataFrame(temp_results),
        pd.DataFrame(replicate_records),
        pd.DataFrame(dynamics_records),
    )

def compute_additional_reduction(results_df, target_coverage=0.50):
    """
    Compute Delta_m(s) = 100 * (Y_random - Y_s) / Y0.
    Y0 is the mean no-vaccination outcome from the Random-labelled
    coverage-zero simulation for the same scenario, VES_0 and R0.
    """
    rows = []
    key_cols = ["Scenario", "VES_0", "R0"]

    for keys, group in results_df.groupby(key_cols):
        key_dict = dict(zip(key_cols, keys))

        no_vax = group[np.isclose(group["Coverage"], 0.0, atol=0.01)]
        target = group[np.isclose(group["Coverage"], target_coverage, atol=0.01)]
        random_row = target[target["Strategy"] == "Random"]

        if no_vax.empty or random_row.empty:
            continue

        y0 = no_vax[OUTCOME_COLUMNS].mean(numeric_only=True)
        yr = random_row.iloc[0]

        for _, strat_row in target.iterrows():
            strat = strat_row["Strategy"]
            if strat == "Random":
                continue
            for outcome in OUTCOME_COLUMNS:
                denom = float(y0[outcome])
                if denom == 0 or np.isnan(denom):
                    delta = np.nan
                else:
                    delta = 100.0 * (float(yr[outcome]) - float(strat_row[outcome])) / denom
                rows.append({
                    **key_dict,
                    "Coverage": target_coverage,
                    "Strategy": strat,
                    "Outcome": outcome,
                    "Y0": denom,
                    "Y_random": float(yr[outcome]),
                    "Y_strategy": float(strat_row[outcome]),
                    "Additional Reduction (%)": delta,
                    "Improves Random": float(delta > 0) if not np.isnan(delta) else np.nan,
                })

    return pd.DataFrame(rows)


def summarize_qualitative_patterns(delta_df):
    """Summarize how often each strategy improves over random for each outcome."""
    if delta_df.empty:
        return pd.DataFrame()

    summary = (
        delta_df
        .groupby(["Scenario", "Strategy", "Outcome"], as_index=False)
        .agg(
            N_parameter_combinations=("Additional Reduction (%)", "count"),
            Mean_additional_reduction=("Additional Reduction (%)", "mean"),
            Median_additional_reduction=("Additional Reduction (%)", "median"),
            Fraction_improves_random=("Improves Random", "mean"),
        )
    )
    summary["Fraction_improves_random"] = summary["Fraction_improves_random"].astype(float)
    return summary


def plot_qualitative_summary(summary_df):
    """Create a compact robustness summary panel."""
    if summary_df.empty:
        return

    plot_df = summary_df.copy()
    plot_df["Fraction improves random (%)"] = 100 * plot_df["Fraction_improves_random"]

    g = sns.catplot(
        data=plot_df,
        x="Outcome",
        y="Fraction improves random (%)",
        hue="Strategy",
        col="Scenario",
        kind="bar",
        height=4,
        aspect=1.15,
        palette="deep"
    )
    g.set_axis_labels("Outcome", "Parameter combinations with advantage (%)")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "Robustness_Qualitative_Summary.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("INITIATING DISEASE-LIKE ROBUSTNESS EXECUTION")
    print(f"{'='*60}")

    tasks = []
    for scenario_name, params in SCENARIOS.items():
        for ves0 in params["VES_0_LIST"]:
            for r0 in params["R0_LIST"]:
                tasks.append((scenario_name, ves0, r0))

    print(f"[INFO] Total tasks: {len(tasks)}")
    print(f"[INFO] Strategies: {strategies_list}")
    print(f"[INFO] Coverage levels: {COVERAGE_STEPS.tolist()}")
    print(f"[INFO] MC runs per setting: {MC_RUNS_PER_POINT}")
    print(f"[INFO] Save dynamics curves: {SAVE_DYNAMICS_CURVES}")

    all_results_df_list = []
    all_replicate_df_list = []
    all_dynamics_df_list = []

    with ProcessPoolExecutor(max_workers=MAX_CORES) as executor:
        futures = {
            executor.submit(run_single_scenario, scenario_name, ves0, r0): (scenario_name, ves0, r0)
            for scenario_name, ves0, r0 in tasks
        }

        for future in as_completed(futures):
            scenario_name, ves0, r0 = futures[future]
            try:
                result_df, replicate_df, dynamics_df = future.result()
                all_results_df_list.append(result_df)
                if replicate_df is not None and not replicate_df.empty:
                    all_replicate_df_list.append(replicate_df)
                if dynamics_df is not None and not dynamics_df.empty:
                    all_dynamics_df_list.append(dynamics_df)
            except Exception as e:
                print(
                    f"[ERROR] Task Scenario={scenario_name}, VES0={ves0}, R0={r0} generated an exception: {e}",
                    flush=True
                )

    if not all_results_df_list:
        raise RuntimeError("No simulation results were generated.")

    final_all_results_df = pd.concat(all_results_df_list, ignore_index=True)
    excel_name = str(OUTPUT_DIR / "Simulation_Results_Disease_Like_Robustness.xlsx")
    final_all_results_df.to_excel(excel_name, index=False)
    print(f"[SAVED] Raw robustness simulation summaries saved to {excel_name}")

    if all_replicate_df_list:
        final_replicate_df = pd.concat(all_replicate_df_list, ignore_index=True)
        replicate_name = str(OUTPUT_DIR / "Disease_Robustness_Replicate_Outcomes.csv.gz")
        final_replicate_df.to_csv(replicate_name, index=False, compression="gzip")
        print(f"[SAVED] Replicate-level outcomes and age-group counts saved to {replicate_name}")

    if SAVE_DYNAMICS_CURVES and all_dynamics_df_list:
        final_dynamics_df = pd.concat(all_dynamics_df_list, ignore_index=True)
        dynamics_name = str(OUTPUT_DIR / "Dynamics_Curves_Disease_Like_Robustness.xlsx")
        final_dynamics_df.to_excel(dynamics_name, index=False)
        print(f"[SAVED] Dynamics curve data saved to {dynamics_name}")

    delta_df = compute_additional_reduction(final_all_results_df, target_coverage=0.50)
    summary_df = summarize_qualitative_patterns(delta_df)

    scenario_parameter_rows = []
    for scenario_name, params in SCENARIOS.items():
        scenario_parameter_rows.append({
            "Scenario": scenario_name,
            **{k: v for k, v in params.items() if k not in ["VES_0_LIST", "R0_LIST"]},
            "VES_0_LIST": ",".join(map(str, params["VES_0_LIST"])),
            "R0_LIST": ",".join(map(str, params["R0_LIST"])),
            "External Prevalence": EXTERNAL_PREVALENCE,
            "IFR_Structure": "COVID-like baseline age-specific IFR",
            "MC Runs": MC_RUNS_PER_POINT,
            "Simulation Days": SIM_DAYS_STRATEGY,
            "Initial Infectious Nodes": NUM_SEEDS_PER_RUN,
        })
    scenario_parameters_df = pd.DataFrame(scenario_parameter_rows)
    age_population_df = pd.DataFrame([AGE_GROUP_POPULATION])

    with pd.ExcelWriter(str(OUTPUT_DIR / "Robustness_Additional_Reduction_Summary.xlsx")) as writer:
        delta_df.to_excel(writer, sheet_name="Additional_Reduction", index=False)
        summary_df.to_excel(writer, sheet_name="Qualitative_Summary", index=False)
        scenario_parameters_df.to_excel(writer, sheet_name="Scenario_Parameters", index=False)
        age_population_df.to_excel(writer, sheet_name="Age_Group_Population", index=False)

    print("[SAVED] Robustness_Additional_Reduction_Summary.xlsx")

    # Compact summary plots are retained; per-scenario dynamics PNGs are disabled.
    plot_qualitative_summary(summary_df)
    print("[SAVED] Robustness_Qualitative_Summary.png")

    fig_leg = plt.figure(figsize=(8, 1.8))
    handles = [plt.Line2D([0], [0], color=strat_colors[s], lw=3) for s in plot_strategies]
    fig_leg.legend(handles, plot_strategies, loc='center', ncol=len(plot_strategies), frameon=False, fontsize=11)
    plt.axis('off')
    plt.savefig(str(OUTPUT_DIR / "Standalone_Legend_Robustness_Strategies.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("[DONE] Disease-like robustness analysis finished.")
