"""Network-structure sensitivity experiments for Figure S8."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import matplotlib
matplotlib.use('Agg')
import hashlib
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import eigsh
current_path = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = '20231012.xlsx'
DATA_PATH = os.environ.get('LEAKY_DATA', str(Path(__file__).resolve().parents[1] / 'data' / '20231012.xlsx'))
SHEET_ID = 0
MAX_CORES = int(os.environ.get('LEAKY_WORKERS', min(20, max(1, (os.cpu_count() or 2) - 1))))
RNG_SEED = 42
SCENARIO_NAME = 'COVID_like_baseline'
SCENARIO_PARAMS = {'PROB_E_TO_I': 0.2, 'PROB_I_TO_S': 0.142, 'VAX_FULL_DAYS': 100, 'VAX_MID_DAYS': 180, 'VAX_MID_MULT': 0.7, 'VAX_LONG_MULT': 0.4, 'NAT_EFF_MAX': 0.652, 'NAT_EFF_MIN': 0.247, 'NAT_FULL_DAYS': 90, 'NAT_WANE_DAYS': 365}
R0_LIST = [float(x) for x in os.environ['LEAKY_R0'].split(',')] if 'LEAKY_R0' in os.environ else [1.0, 2.0, 4.0, 8.0]
VES_0_LIST = [float(x) for x in os.environ['LEAKY_VE'].split(',')] if 'LEAKY_VE' in os.environ else [0.2, 0.5, 0.8]
COVERAGE_STEPS = np.array([float(x) for x in os.environ['LEAKY_COVERAGE'].split(',')]) if 'LEAKY_COVERAGE' in os.environ else np.array([0.0, 0.5])
MC_RUNS_PER_POINT = int(os.environ.get('LEAKY_RUNS', 100))
SIM_DAYS_STRATEGY = int(os.environ.get('LEAKY_DAYS', 500))
USE_COMMON_RANDOM_NUMBERS = False
STRICT_OUTPUT_VALIDATION = True
ALLOW_DUPLICATE_HOUSEHOLD_MEMBER_KEYS = True
NUM_SEEDS_PER_RUN = int(os.environ.get('LEAKY_SEEDS', 10))
NETWORK_REPLICATES_MAIN = 10
NETWORK_REPLICATES_SI = 5
SAVE_DYNAMICS_CURVES = False
SAVE_REPLICATE_OUTCOMES = True
SAVE_AGE_GROUP_COUNTS = True
HH_CONTACT_HOURS = 12.0
SOCIAL_ENCOUNTER_DURATION = 0.25
INTERNAL_SOCIAL_RATIO = 0.3
VIRTUAL_CONTACT_MAPPING = {'a': 3.0, 'b': 10.0, 'c': 20.0, 'd': 30.0}
EXTERNAL_PREVALENCE = 0.005
# Effective contact count, not contacts per hour.
EXTERNAL_SOCIAL_CONTACT_COUNT = 4.0
strategies_list = ['Random', 'High Strength (Node)', 'Priority: Oldest First']
plot_strategies = strategies_list.copy()
OUTCOME_COLUMNS = ['Final Infection Proportion', 'Epidemic Peak Size', 'Expected Deaths']
MAIN_TEXT_NETWORK_TYPES = ['Empirical', 'CommunityResampled', 'SmallWorldCommunity', 'HouseholdAttenuatedCompensated']
SI_NETWORK_TYPES = ['ERRandom', 'BAScaleFree', 'CompletelyRandomized']
ALL_NETWORK_TYPES = MAIN_TEXT_NETWORK_TYPES + SI_NETWORK_TYPES
IFR_MAP = {15: 6.95e-05, 24: 0.000309, 34.5: 0.000844, 44.5: 0.00161, 54.5: 0.00595, 64.5: 0.0193, 75: 0.0428}
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

def make_stable_seed(*items, base_seed=RNG_SEED):
    key = '|'.join(map(str, items))
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()
    return int((base_seed + int(digest[:8], 16)) % (2 ** 32 - 1))

def descending_order_with_explicit_ties(values):
    values = np.asarray(values, dtype=float)
    node_ids = np.arange(values.size)
    return np.lexsort((-node_ids, -values))

def summarize_metric(values, prefix):
    arr = np.asarray(values, dtype=float)
    n_obs = len(arr)
    mean_val = float(np.mean(arr)) if n_obs else np.nan
    sd_val = float(np.std(arr, ddof=1)) if n_obs > 1 else 0.0
    se_val = sd_val / np.sqrt(n_obs) if n_obs > 0 else np.nan
    return {prefix: mean_val, '{} SD'.format(prefix): sd_val, '{} SE'.format(prefix): se_val, '{} CI95 Lower'.format(prefix): mean_val - 1.96 * se_val if n_obs > 0 else np.nan, '{} CI95 Upper'.format(prefix): mean_val + 1.96 * se_val if n_obs > 0 else np.nan}

def get_age_group_infection_counts(ever_infected_mask, vax_mask):
    counts = {}
    for age in AGE_GROUP_VALUES:
        label = AGE_GROUP_LABELS[age]
        age_mask = np.isclose(AGE_NUM_VALUES, age)
        infected_age = ever_infected_mask & age_mask
        counts['Infected_Age_{}'.format(label)] = int(np.sum(infected_age))
        counts['Vaccinated_Infected_Age_{}'.format(label)] = int(np.sum(infected_age & vax_mask))
        counts['Unvaccinated_Infected_Age_{}'.format(label)] = int(np.sum(infected_age & ~vax_mask))
    return counts

def _normalize_col(s):
    return str(s).replace('\ufeff', '').strip().replace(' ', '').replace('\uff08', '(').replace('\uff09', ')').replace('/', '')

def pick_column(df, targets):
    norm_map = {col: _normalize_col(col) for col in df.columns}
    wanted_norm = [_normalize_col(t) for t in targets]
    for col, ncol in norm_map.items():
        if ncol in wanted_norm:
            return col
    for col, ncol in norm_map.items():
        if any((w in ncol for w in wanted_norm)) or any((ncol in w for w in wanted_norm)):
            return col
    raise KeyError('Column not found: {}'.format(targets))
FREQ_MAP = {'\u65e0': 0, '<1\u6b21': 0.5, '\uff1c1\u6b21': 0.5, '\uff1c2\u6b21': 0.5, '1-2\u6b21': 1.5, '1-3\u6b21': 0.5, '3-4\u6b21': 3.5, '\u22655\u6b21': 5.5, 'a': 5.5, 'b': 3.5, 'c': 1.5, 'd': 0.5}
DUR_MAP = {'\u65e0': 0, '2\u5c0f\u65f6\u4ee5\u5185': 1, '3\u5c0f\u65f6\u4ee5\u5185': 0.0, '2-5\u5c0f\u65f6': 3.5, '5\u5c0f\u65f6\u4ee5\u4e0a': 6, 'a': 1, 'b': 3.5, 'c': 6}

def _map_freq(x):
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if s in FREQ_MAP:
        return FREQ_MAP[s]
    if '5\u6b21' in s:
        return 5.5
    raise ValueError('Unrecognised trip-frequency code: {!r}'.format(x))

def _map_dur(x):
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if s in DUR_MAP:
        return DUR_MAP[s]
    if '2-5' in s:
        return 3.5
    raise ValueError('Unrecognised trip-duration code: {!r}'.format(x))

def _infer_employed(s):
    s = str(s).strip()
    return s in ['b', 'B', 'd', 'D'] or '\u5de5' in s or '\u73ed' in s

def _infer_student(s):
    s = str(s).strip()
    return '\u5b66' in s or s in ['a', 'A']

def _map_contact_count(x):
    if pd.isna(x):
        return 0.0
    s = str(x).lower().strip()
    if s in {'\u65e0', 'none', '0', '0.0'}:
        return 0.0
    if s in VIRTUAL_CONTACT_MAPPING:
        return VIRTUAL_CONTACT_MAPPING[s]
    raise ValueError('Unrecognised work/school contact-count code: {!r}'.format(x))

def gini_coefficient(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    if np.min(x) < 0:
        x = x - np.min(x)
    total = float(np.sum(x))
    if total == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = len(x_sorted)
    ranks = np.arange(1, n + 1, dtype=float)
    return float(2.0 * np.sum(ranks * x_sorted) / (n * total) - (n + 1.0) / n)

def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])

def calculate_expected_deaths(infected_indices, age_series):
    deaths = 0.0
    ages = age_series.iloc[infected_indices].values
    available_ages = np.array(list(IFR_MAP.keys()))
    for age in ages:
        idx = np.abs(available_ages - age).argmin()
        closest_age = available_ages[idx]
        deaths += IFR_MAP[closest_age]
    return float(deaths)

def build_sparse_adj_matrix(graph, n_nodes):
    if hasattr(nx, 'to_scipy_sparse_array'):
        return nx.to_scipy_sparse_array(graph, nodelist=range(n_nodes), weight='weight', format='csr', dtype=float)
    return nx.to_scipy_sparse_matrix(graph, nodelist=range(n_nodes), weight='weight', format='csr', dtype=float)

def spectral_radius_weighted(graph, n_nodes):
    adj_mat = build_sparse_adj_matrix(graph, n_nodes)
    try:
        eigenvalues, _ = eigsh(adj_mat, k=1, which='LA')
        sr = float(eigenvalues[0])
        if sr <= 0:
            return np.nan
        return sr
    except Exception as e:
        print('[WARNING] Eigenvalue calculation failed: {}'.format(e), flush=True)
        return np.nan

def calibrate_beta(target_r0, gamma, graph, n_nodes):
    sr = spectral_radius_weighted(graph, n_nodes)
    if not np.isfinite(sr) or sr <= 0:
        raise RuntimeError('Invalid weighted spectral radius {}; calibration aborted.'.format(sr))
    return target_r0 * gamma * 24.0 / sr
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError('Data file not found: {}'.format(DATA_PATH))
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_ID)
N = len(df)
print('[INFO] Loaded N = {}'.format(N), flush=True)
COL_COMMUNITY = pick_column(df, ['\u5c0f\u533a\u540d\u79f0', '\u5c0f\u533a', '\u793e\u533a\u540d\u79f0', '\u793e\u533a'])
COL_ADDRESS = pick_column(df, ['\u4f4f\u6237\u8be6\u7ec6\u5730\u5740', '\u4f4f\u5740'])
COL_HH_INDEX = pick_column(df, ['\u672c\u6237\u7b2c\u4efd'])
COL_WORK_STATUS = pick_column(df, ['\u5de5\u4f5c\u72b6\u6001'])
COL_CONTACT_COUNT = pick_column(df, ['\u4eba\u5458\u6570\u91cf', '\u5b66\u4e60\u573a\u6240\u7684\u4eba\u5458\u6570\u91cf'])
COL_TRIP_FREQ = ['\u9891\u7387{}'.format(i) for i in range(1, 6)]
COL_TRIP_DUR = ['\u65f6\u957f{}'.format(i) for i in range(1, 6)]
COL_AGE = pick_column(df, ['\u5e74\u9f84'])
df['_community_clean'] = df[COL_COMMUNITY].astype(str).str.strip()
df['_address_clean'] = df[COL_ADDRESS].astype(str).str.strip()
df['_hh_index_clean'] = df[COL_HH_INDEX].astype(str).str.strip()
df['_household_key'] = df['_community_clean'] + '|' + df['_address_clean']
df['_household_member_key'] = df['_household_key'] + '|' + df['_hh_index_clean']
household_sizes = df.groupby('_household_key').size()
duplicated_member_mask = df['_household_member_key'].duplicated(keep=False)
print('[HOUSEHOLD CHECK] Unique households: {}; households with >=2 respondents: {}; duplicate household-member rows: {}'.format(df['_household_key'].nunique(), int((household_sizes >= 2).sum()), int(duplicated_member_mask.sum())), flush=True)
if duplicated_member_mask.any() and (not ALLOW_DUPLICATE_HOUSEHOLD_MEMBER_KEYS):
    duplicate_keys = df.loc[duplicated_member_mask, '_household_member_key'].value_counts().head(20).to_dict()
    raise ValueError('Duplicated community + address + household-member-index keys were detected. Resolve the source data or explicitly set ALLOW_DUPLICATE_HOUSEHOLD_MEMBER_KEYS=True after justification. Examples: {}'.format(duplicate_keys))
age_map_dict = {'a': 15, 'b': 24, 'c': 34.5, 'd': 44.5, 'e': 54.5, 'f': 64.5, 'g': 75}
df['age_num'] = df[COL_AGE].astype(str).str.strip().map(age_map_dict).fillna(45.0)
AGE_NUM_VALUES = df['age_num'].values.astype(float)
AGE_GROUP_VALUES = np.array(sorted(np.unique(AGE_NUM_VALUES)), dtype=float)
AGE_GROUP_LABELS = {age: str(age).replace('.', '_') for age in AGE_GROUP_VALUES}
AGE_GROUP_POPULATION = {f'Population_Age_{AGE_GROUP_LABELS[age]}': int(np.sum(np.isclose(AGE_NUM_VALUES, age))) for age in AGE_GROUP_VALUES}
daily_hours_nonfixed = np.zeros(N)
for k in range(5):
    if COL_TRIP_FREQ[k] in df.columns:
        fk = df[COL_TRIP_FREQ[k]].map(_map_freq).fillna(0).values
        dk = df[COL_TRIP_DUR[k]].map(_map_dur).fillna(0).values
        daily_hours_nonfixed += fk * dk / 7.0
daily_hours_fixed = np.zeros(N)
is_key_group = np.zeros(N, dtype=bool)
status_arr = df[COL_WORK_STATUS].values
for i in range(N):
    if _infer_employed(status_arr[i]) or _infer_student(status_arr[i]):
        daily_hours_fixed[i] = 8.0
        is_key_group[i] = True
mobility_index = np.clip(daily_hours_fixed + daily_hours_nonfixed, 0, 12.0)
virtual_contact_counts = df[COL_CONTACT_COUNT].map(_map_contact_count).fillna(0).values
strength_vir_work = virtual_contact_counts * daily_hours_fixed
external_social_time = daily_hours_nonfixed * (1.0 - INTERNAL_SOCIAL_RATIO)
strength_vir_social = external_social_time * EXTERNAL_SOCIAL_CONTACT_COUNT
strength_virtual = strength_vir_work + strength_vir_social
house2members = defaultdict(list)
for i in range(N):
    house2members[df.iloc[i]['_household_key']].append(i)
node_to_house = {}
for h, members in house2members.items():
    for node in members:
        node_to_house[node] = h
internal_social_time = daily_hours_nonfixed * INTERNAL_SOCIAL_RATIO
community_encounter_counts = np.array([int(round(t / SOCIAL_ENCOUNTER_DURATION)) if t > 0 else 0 for t in internal_social_time], dtype=int)

def add_household_edges(graph, home_scale=1.0):
    for members in house2members.values():
        m = len(members)
        if m <= 1:
            continue
        w = home_scale * HH_CONTACT_HOURS / (m - 1)
        for a in range(m):
            for b in range(a + 1, m):
                graph.add_edge(members[a], members[b], weight=w, layer='home')

def add_random_budget_community_edges(graph, rng, weight_scale=1.0):
    nodes_indices = np.arange(N)
    for i in range(N):
        n_encounters = int(community_encounter_counts[i])
        if n_encounters <= 0:
            continue
        candidates = nodes_indices[nodes_indices != i]
        if len(candidates) == 0:
            continue
        tgts = rng.choice(candidates, size=min(n_encounters, N - 1), replace=False)
        w = SOCIAL_ENCOUNTER_DURATION * weight_scale
        for t in tgts:
            if graph.has_edge(i, int(t)):
                graph[i][int(t)]['weight'] += w
                if graph[i][int(t)].get('layer') != 'home':
                    graph[i][int(t)]['layer'] = 'community_random'
            else:
                graph.add_edge(i, int(t), weight=w, layer='community_random')

def scale_community_edges_to_target_internal_strength(graph, target_sum_internal):
    current_sum_internal = sum(dict(graph.degree(weight='weight')).values())
    if current_sum_internal <= 0:
        return
    home_sum = 0.0
    community_edges = []
    community_weight_sum_degree = 0.0
    for u, v, data in graph.edges(data=True):
        w = float(data.get('weight', 1.0))
        if data.get('layer') == 'home':
            home_sum += 2.0 * w
        else:
            community_edges.append((u, v))
            community_weight_sum_degree += 2.0 * w
    needed_community_degree_sum = target_sum_internal - home_sum
    if community_weight_sum_degree <= 0 or needed_community_degree_sum <= 0:
        return
    factor = needed_community_degree_sum / community_weight_sum_degree
    for u, v in community_edges:
        graph[u][v]['weight'] *= factor

def build_reference_empirical_network(seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    add_household_edges(graph, home_scale=1.0)
    add_random_budget_community_edges(graph, rng, weight_scale=1.0)
    return graph

def sample_weight_list_from_graph(graph):
    weights_all = []
    weights_nonhome = []
    for _, _, data in graph.edges(data=True):
        w = float(data.get('weight', 1.0))
        weights_all.append(w)
        if data.get('layer') != 'home':
            weights_nonhome.append(w)
    if len(weights_nonhome) == 0:
        weights_nonhome = [SOCIAL_ENCOUNTER_DURATION]
    if len(weights_all) == 0:
        weights_all = [SOCIAL_ENCOUNTER_DURATION]
    return (np.array(weights_all, dtype=float), np.array(weights_nonhome, dtype=float))
REFERENCE_EMPIRICAL_GRAPH = build_reference_empirical_network(RNG_SEED)
REFERENCE_INTERNAL_STRENGTH = np.array([REFERENCE_EMPIRICAL_GRAPH.degree(n, weight='weight') for n in range(N)])
REFERENCE_SUM_INTERNAL_STRENGTH = float(np.sum(REFERENCE_INTERNAL_STRENGTH))
REFERENCE_NUM_EDGES = REFERENCE_EMPIRICAL_GRAPH.number_of_edges()
REFERENCE_WEIGHTS_ALL, REFERENCE_WEIGHTS_NONHOME = sample_weight_list_from_graph(REFERENCE_EMPIRICAL_GRAPH)
REFERENCE_NUM_NONHOME_EDGES = sum((1 for _, _, data in REFERENCE_EMPIRICAL_GRAPH.edges(data=True) if data.get('layer') != 'home'))
HOUSEHOLD_EDGE_SET = set()
for members in house2members.values():
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            u, v = sorted((members[a], members[b]))
            HOUSEHOLD_EDGE_SET.add((u, v))
print('[NETWORK] Reference empirical network built. Edges={}, Community edges={}, Avg internal strength={:.2f}'.format(REFERENCE_NUM_EDGES, REFERENCE_NUM_NONHOME_EDGES, REFERENCE_INTERNAL_STRENGTH.mean()), flush=True)

def add_weighted_edges_from_list(graph, edge_list, weights, layer='community_random'):
    for idx, (u, v) in enumerate(edge_list):
        w = float(weights[idx % len(weights)])
        if u == v:
            continue
        if graph.has_edge(u, v):
            graph[u][v]['weight'] += w
            if graph[u][v].get('layer') != 'home':
                graph[u][v]['layer'] = layer
        else:
            graph.add_edge(u, v, weight=w, layer=layer)

def random_unique_edges(rng, num_edges, exclude_existing=None):
    if exclude_existing is None:
        exclude_existing = set()
    edges = set()
    attempts = 0
    max_attempts = max(10000, num_edges * 50)
    while len(edges) < num_edges and attempts < max_attempts:
        u = int(rng.randint(0, N))
        v = int(rng.randint(0, N))
        if u == v:
            attempts += 1
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edges or (a, b) in exclude_existing:
            attempts += 1
            continue
        edges.add((a, b))
        attempts += 1
    if len(edges) != num_edges:
        raise RuntimeError('Could not generate the requested number of unique random edges: requested {}, generated {}.'.format(num_edges, len(edges)))
    return list(edges)

def build_small_world_community_graph(rng):
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    add_household_edges(graph, home_scale=1.0)
    desired_edges = max(1, int(REFERENCE_NUM_NONHOME_EDGES))
    k = max(2, int(round(2.0 * desired_edges / float(N))))
    if k % 2 == 1:
        k += 1
    k = min(k, N - 1 if (N - 1) % 2 == 0 else N - 2)
    if k < 2:
        k = 2
    ws_seed = int(rng.randint(0, 2 ** 31 - 1))
    ws = nx.watts_strogatz_graph(N, k, 0.1, seed=ws_seed)
    permutation = rng.permutation(N)
    mapping = {old: int(permutation[old]) for old in range(N)}
    ws = nx.relabel_nodes(ws, mapping, copy=True)
    candidate_edges = []
    for u, v in ws.edges():
        if u == v:
            continue
        edge = (min(u, v), max(u, v))
        if edge in HOUSEHOLD_EDGE_SET:
            continue
        candidate_edges.append(edge)
    rng.shuffle(candidate_edges)
    chosen = candidate_edges[:desired_edges]
    if len(chosen) < desired_edges:
        existing = set(chosen) | HOUSEHOLD_EDGE_SET
        extra = random_unique_edges(rng, desired_edges - len(chosen), exclude_existing=existing)
        chosen.extend(extra)
    weights = rng.choice(REFERENCE_WEIGHTS_NONHOME, size=len(chosen), replace=True)
    add_weighted_edges_from_list(graph, chosen, weights, layer='small_world_community')
    scale_community_edges_to_target_internal_strength(graph, REFERENCE_SUM_INTERNAL_STRENGTH)
    return graph

def build_er_random_graph(rng):
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    edges = random_unique_edges(rng, REFERENCE_NUM_EDGES)
    weights = rng.choice(REFERENCE_WEIGHTS_ALL, size=len(edges), replace=True)
    add_weighted_edges_from_list(graph, edges, weights, layer='er_random')
    scale_community_edges_to_target_internal_strength(graph, REFERENCE_SUM_INTERNAL_STRENGTH)
    return graph

def build_ba_scale_free_graph(rng):
    target_edges = REFERENCE_NUM_EDGES
    m = max(1, int(round(target_edges / float(N))))
    m = min(m, max(1, N - 1))
    ba_seed = int(rng.randint(0, 2 ** 31 - 1))
    ba = nx.barabasi_albert_graph(N, m, seed=ba_seed)
    permutation = rng.permutation(N)
    mapping = {old: int(permutation[old]) for old in range(N)}
    ba = nx.relabel_nodes(ba, mapping, copy=True)
    edges = list(ba.edges())
    rng.shuffle(edges)
    if len(edges) > target_edges:
        edges = edges[:target_edges]
    elif len(edges) < target_edges:
        existing = set(((min(u, v), max(u, v)) for u, v in edges))
        edges.extend(random_unique_edges(rng, target_edges - len(edges), exclude_existing=existing))
    weights = rng.choice(REFERENCE_WEIGHTS_ALL, size=len(edges), replace=True)
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    add_weighted_edges_from_list(graph, edges, weights, layer='ba_scale_free')
    scale_community_edges_to_target_internal_strength(graph, REFERENCE_SUM_INTERNAL_STRENGTH)
    return graph

def build_completely_randomized_graph(rng):
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    edges = random_unique_edges(rng, REFERENCE_NUM_EDGES)
    weights = rng.permutation(REFERENCE_WEIGHTS_ALL)
    if len(weights) < len(edges):
        weights = rng.choice(REFERENCE_WEIGHTS_ALL, size=len(edges), replace=True)
    add_weighted_edges_from_list(graph, edges, weights, layer='completely_randomized')
    scale_community_edges_to_target_internal_strength(graph, REFERENCE_SUM_INTERNAL_STRENGTH)
    return graph

def validate_network_graph(graph, network_type, replicate_id):
    if graph.number_of_nodes() != N:
        raise RuntimeError('Network {} replicate {} has {} nodes; expected {}.'.format(network_type, replicate_id, graph.number_of_nodes(), N))
    if nx.number_of_selfloops(graph) != 0:
        raise RuntimeError('Network {} replicate {} contains self-loops.'.format(network_type, replicate_id))
    weights = np.array([float(data.get('weight', np.nan)) for _, _, data in graph.edges(data=True)], dtype=float)
    if weights.size == 0 or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError('Network {} replicate {} contains missing, non-finite or non-positive weights.'.format(network_type, replicate_id))

def build_network_by_type(network_type, replicate_id):
    seed = make_stable_seed('network', network_type, replicate_id)
    rng = np.random.RandomState(seed)
    if network_type == 'Empirical':
        graph = build_reference_empirical_network(RNG_SEED)
    elif network_type == 'CommunityResampled':
        graph = nx.Graph()
        graph.add_nodes_from(range(N))
        add_household_edges(graph, home_scale=1.0)
        add_random_budget_community_edges(graph, rng, weight_scale=1.0)
    elif network_type == 'SmallWorldCommunity':
        graph = build_small_world_community_graph(rng)
    elif network_type == 'HouseholdAttenuatedCompensated':
        graph = nx.Graph()
        graph.add_nodes_from(range(N))
        add_household_edges(graph, home_scale=0.5)
        add_random_budget_community_edges(graph, rng, weight_scale=1.0)
        scale_community_edges_to_target_internal_strength(graph, REFERENCE_SUM_INTERNAL_STRENGTH)
    elif network_type == 'ERRandom':
        graph = build_er_random_graph(rng)
    elif network_type == 'BAScaleFree':
        graph = build_ba_scale_free_graph(rng)
    elif network_type == 'CompletelyRandomized':
        graph = build_completely_randomized_graph(rng)
    else:
        raise ValueError('Unknown network type: {}'.format(network_type))
    validate_network_graph(graph, network_type, replicate_id)
    return make_network_context(network_type, replicate_id, graph)

def make_network_context(network_type, replicate_id, graph):
    adj_matrix = build_sparse_adj_matrix(graph, N)
    strength_internal = np.array([graph.degree(n, weight='weight') for n in range(N)], dtype=float)
    total_node_strength = strength_internal + strength_virtual
    avg_neighbor_strength = np.zeros(N, dtype=float)
    for i in range(N):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) > 0:
            avg_neighbor_strength[i] = np.mean([total_node_strength[nbr] for nbr in neighbors])
    return {'Network_Type': network_type, 'Network_Replicate': replicate_id, 'G': graph, 'adj_matrix': adj_matrix, 'strength_internal': strength_internal, 'total_node_strength': total_node_strength, 'avg_neighbor_strength': avg_neighbor_strength}

def compute_network_diagnostics(context):
    graph = context['G']
    strength_internal = context['strength_internal']
    total_node_strength = context['total_node_strength']
    degree_arr = np.array([graph.degree(n) for n in range(N)], dtype=float)
    n_edges = graph.number_of_edges()
    home_edges = 0
    community_edges = 0
    total_weight = 0.0
    home_weight = 0.0
    community_weight = 0.0
    for _, _, data in graph.edges(data=True):
        w = float(data.get('weight', 1.0))
        total_weight += w
        if data.get('layer') == 'home':
            home_edges += 1
            home_weight += w
        else:
            community_edges += 1
            community_weight += w
    try:
        mean_clustering = float(nx.average_clustering(graph, weight='weight'))
    except Exception:
        mean_clustering = np.nan
    try:
        assortativity = float(nx.degree_assortativity_coefficient(graph))
    except Exception:
        assortativity = np.nan
    try:
        components = list(nx.connected_components(graph))
        n_components = len(components)
        largest_cc = max(components, key=len)
        giant_component_fraction = len(largest_cc) / float(N)
        if len(largest_cc) > 1 and len(largest_cc) <= 3000:
            avg_shortest_path_gc = float(nx.average_shortest_path_length(graph.subgraph(largest_cc)))
        else:
            avg_shortest_path_gc = np.nan
    except Exception:
        n_components = np.nan
        giant_component_fraction = np.nan
        avg_shortest_path_gc = np.nan
    sr = spectral_radius_weighted(graph, N)
    return {'Network_Type': context['Network_Type'], 'Network_Replicate': context['Network_Replicate'], 'Number_of_edges': n_edges, 'Home_edge_fraction': home_edges / n_edges if n_edges > 0 else np.nan, 'Community_edge_fraction': community_edges / n_edges if n_edges > 0 else np.nan, 'Home_weight_fraction': home_weight / total_weight if total_weight > 0 else np.nan, 'Community_weight_fraction': community_weight / total_weight if total_weight > 0 else np.nan, 'Mean_degree': float(np.mean(degree_arr)), 'Mean_internal_strength': float(np.mean(strength_internal)), 'Mean_total_strength': float(np.mean(total_node_strength)), 'Gini_total_strength': gini_coefficient(total_node_strength), 'Weighted_spectral_radius': sr, 'Mean_weighted_clustering': mean_clustering, 'Degree_assortativity': assortativity, 'Number_connected_components': n_components, 'Giant_component_fraction': giant_component_fraction, 'Average_shortest_path_largest_component': avg_shortest_path_gc, 'Age_strength_correlation': safe_corr(df['age_num'].values, total_node_strength)}

def get_vaccinated_mask(strategy, coverage, context, rng):
    num_vax = int(coverage * N)
    if num_vax == 0:
        return np.zeros(N, dtype=bool)
    if strategy == 'Random':
        indices = rng.choice(N, num_vax, replace=False)
    elif strategy == 'Priority: Oldest First':
        indices = descending_order_with_explicit_ties(df['age_num'].values)[:num_vax]
    elif strategy == 'High Strength (Node)':
        indices = descending_order_with_explicit_ties(context['total_node_strength'])[:num_vax]
    else:
        raise ValueError('Strategy not implemented in network robustness run: {}'.format(strategy))
    mask = np.zeros(N, dtype=bool)
    mask[indices] = True
    return mask

def run_strategy_simulation(beta_val, vax_mask, ves_0, context, simulation_seed):
    rng = np.random.default_rng(simulation_seed)
    state = np.zeros(N, dtype=int)
    days_since_rec = np.full(N, -1, dtype=int)
    ever_infected = np.zeros(N, dtype=bool)
    adj_matrix = context['adj_matrix']
    lambda_external_const = np.where(is_key_group, beta_val * strength_vir_work / 24.0 * EXTERNAL_PREVALENCE, 0.0) + np.where(daily_hours_nonfixed > 0, beta_val * strength_vir_social / 24.0 * EXTERNAL_PREVALENCE, 0.0)
    seeds = rng.choice(N, size=NUM_SEEDS_PER_RUN, replace=False)
    state[seeds] = 2
    ever_infected[seeds] = True
    daily_infected_counts = []
    for d in range(1, SIM_DAYS_STRATEGY + 1):
        new_state = state.copy()
        r_val = rng.random(N)
        r_rec = rng.random(N)
        if d <= SCENARIO_PARAMS['VAX_FULL_DAYS']:
            current_ve = ves_0
        elif d <= SCENARIO_PARAMS['VAX_MID_DAYS']:
            current_ve = SCENARIO_PARAMS['VAX_MID_MULT'] * ves_0
        else:
            current_ve = SCENARIO_PARAMS['VAX_LONG_MULT'] * ves_0
        vax_susc_mult = np.where(vax_mask, 1.0 - current_ve, 1.0)
        nat_susc_mult = np.ones(N)
        nat_full_days = SCENARIO_PARAMS['NAT_FULL_DAYS']
        nat_wane_days = SCENARIO_PARAMS['NAT_WANE_DAYS']
        nat_eff_max = SCENARIO_PARAMS['NAT_EFF_MAX']
        nat_eff_min = SCENARIO_PARAMS['NAT_EFF_MIN']
        mask_full = (days_since_rec >= 0) & (days_since_rec <= nat_full_days)
        mask_wane = (days_since_rec > nat_full_days) & (days_since_rec <= nat_wane_days)
        mask_long = days_since_rec > nat_wane_days
        nat_susc_mult[mask_full] = 1.0 - nat_eff_max
        if np.any(mask_wane):
            decay_ratio = (days_since_rec[mask_wane] - nat_full_days) / max(nat_wane_days - nat_full_days, 1)
            current_eff = nat_eff_max - (nat_eff_max - nat_eff_min) * decay_ratio
            nat_susc_mult[mask_wane] = 1.0 - current_eff
        nat_susc_mult[mask_long] = 1.0 - nat_eff_min
        total_susc_mult = vax_susc_mult * nat_susc_mult
        sus_indices = np.where(state == 0)[0]
        if len(sus_indices) > 0:
            I_vector = (state == 2).astype(float)
            lambda_internal = beta_val / 24.0 * adj_matrix.dot(I_vector)
            lambda_total_all = lambda_internal + lambda_external_const
            lambda_sus = lambda_total_all[sus_indices]
            susc_sus = total_susc_mult[sus_indices]
            prob_sus = 1.0 - np.exp(-susc_sus * lambda_sus)
            infected_mask = (prob_sus > 0) & (r_val[sus_indices] < prob_sus)
            newly_infected_indices = sus_indices[infected_mask]
            new_state[newly_infected_indices] = 1
            ever_infected[newly_infected_indices] = True
        exp_indices = np.where(state == 1)[0]
        new_state[exp_indices] = np.where(r_val[exp_indices] < SCENARIO_PARAMS['PROB_E_TO_I'], 2, 1)
        inf_indices = np.where(state == 2)[0]
        recovered = r_rec[inf_indices] < SCENARIO_PARAMS['PROB_I_TO_S']
        new_state[inf_indices[recovered]] = 0
        days_since_rec[inf_indices[recovered]] = 0
        mask_increment = (new_state == 0) & (days_since_rec >= 0)
        days_since_rec[mask_increment] += 1
        days_since_rec[new_state != 0] = -1
        state = new_state
        daily_infected_counts.append(np.sum(state == 2))
    unique_infected_total = int(np.sum(ever_infected))
    daily_infected_counts = np.array(daily_infected_counts)
    peak_sz = float(np.max(daily_infected_counts)) if len(daily_infected_counts) > 0 else 0.0
    peak_tm = float(np.argmax(daily_infected_counts) + 1) if len(daily_infected_counts) > 0 else 0.0
    infected_indices = np.where(ever_infected)[0]
    expected_deaths = calculate_expected_deaths(infected_indices, df['age_num'])
    return (unique_infected_total, peak_sz, peak_tm, expected_deaths, daily_infected_counts, ever_infected)

def run_network_replicate(network_type, replicate_id):
    stable_seed = make_stable_seed('worker', network_type, replicate_id)
    np.random.seed(stable_seed)
    random.seed(stable_seed)
    print('[WORKER START] Network={}, replicate={}'.format(network_type, replicate_id), flush=True)
    start_time = time.time()
    context = build_network_by_type(network_type, replicate_id)
    diagnostics = compute_network_diagnostics(context)
    temp_results = []
    replicate_records = []
    dynamics_records = []
    for ves0 in VES_0_LIST:
        for target_r0 in R0_LIST:
            sum_internal = float(np.sum(context['strength_internal']))
            work_exposure = float(np.sum(strength_vir_work[is_key_group]) * EXTERNAL_PREVALENCE)
            social_exposure = float(np.sum(strength_vir_social[daily_hours_nonfixed > 0]) * EXTERNAL_PREVALENCE)
            sum_external = work_exposure + social_exposure
            internal_ratio = sum_internal / (sum_internal + sum_external + 1e-09)
            adjusted_target_r0 = target_r0 * internal_ratio
            current_beta = calibrate_beta(adjusted_target_r0, SCENARIO_PARAMS['PROB_I_TO_S'], context['G'], N)
            temporal_curves_storage = defaultdict(list)
            for cov_idx, cov in enumerate(COVERAGE_STEPS, start=1):
                print('[PROGRESS] Network={}, rep={}, VES_0={}, R0={}, Coverage {}/{}={:.2f}, Elapsed={:.1f} min'.format(network_type, replicate_id, ves0, target_r0, cov_idx, len(COVERAGE_STEPS), cov, (time.time() - start_time) / 60.0), flush=True)
                strategies_to_run = ['Random'] if np.isclose(cov, 0.0, atol=0.01) else strategies_list
                is_target_point = np.isclose(cov, 0.5, atol=0.01)
                for strat in strategies_to_run:
                    mask_seed = make_stable_seed('vaccination_mask', network_type, replicate_id, ves0, target_r0, float(cov), strat)
                    mask_rng = np.random.default_rng(mask_seed)
                    vax_mask = get_vaccinated_mask(strat, cov, context, mask_rng)
                    actual_num_vax = int(np.sum(vax_mask))
                    actual_coverage = actual_num_vax / float(N)
                    final_inf_prop, peak_sz, peak_tm, exp_deaths = ([], [], [], [])
                    for mc_run in range(1, MC_RUNS_PER_POINT + 1):
                        if np.isclose(cov, 0.0, atol=0.01):
                            simulation_seed = make_stable_seed('epidemic', network_type, replicate_id, target_r0, 0.0, mc_run)
                        else:
                            seed_items = ['epidemic', network_type, replicate_id, ves0, target_r0, float(cov), mc_run]
                            if not USE_COMMON_RANDOM_NUMBERS:
                                seed_items.append(strat)
                            simulation_seed = make_stable_seed(*seed_items)
                        c, p_sz, p_tm, d, daily_curve, ever_infected_mask = run_strategy_simulation(current_beta, vax_mask, ves0, context, simulation_seed)
                        final_value = c / float(N)
                        final_inf_prop.append(final_value)
                        peak_sz.append(p_sz)
                        peak_tm.append(p_tm)
                        exp_deaths.append(d)
                        if SAVE_DYNAMICS_CURVES and is_target_point:
                            temporal_curves_storage[strat].append(daily_curve)
                        if SAVE_REPLICATE_OUTCOMES:
                            replicate_row = {'Scenario': SCENARIO_NAME, 'Network_Type': network_type, 'Network_Replicate': replicate_id, 'VES_0': ves0, 'R0': target_r0, 'Coverage': float(cov), 'Actual Coverage': actual_coverage, 'Actual Vaccinated Count': actual_num_vax, 'Strategy': strat, 'MC_Run': mc_run, 'Simulation Seed': int(simulation_seed), 'Vaccination Mask Seed': int(mask_seed), 'Common Random Numbers': bool(USE_COMMON_RANDOM_NUMBERS), 'Final Infection Proportion': float(final_value), 'Epidemic Peak Size': float(p_sz), 'Peak Time': float(p_tm), 'Expected Deaths': float(d), 'IFR_Structure': 'COVID-like baseline age-specific IFR'}
                            if SAVE_AGE_GROUP_COUNTS:
                                replicate_row.update(get_age_group_infection_counts(ever_infected_mask, vax_mask))
                            replicate_records.append(replicate_row)
                    temp_results.append({'Scenario': SCENARIO_NAME, 'Network_Type': network_type, 'Network_Replicate': replicate_id, 'VES_0': ves0, 'R0': target_r0, 'Coverage': float(cov), 'Actual Coverage': actual_coverage, 'Actual Vaccinated Count': actual_num_vax, 'Strategy': strat, 'Vaccination Mask Seed': int(mask_seed), 'Common Random Numbers': bool(USE_COMMON_RANDOM_NUMBERS), **summarize_metric(final_inf_prop, 'Final Infection Proportion'), **summarize_metric(peak_sz, 'Epidemic Peak Size'), **summarize_metric(peak_tm, 'Peak Time'), **summarize_metric(exp_deaths, 'Expected Deaths'), 'Adjusted_R0_Internal': float(adjusted_target_r0), 'Beta': float(current_beta), 'PROB_E_TO_I': SCENARIO_PARAMS['PROB_E_TO_I'], 'PROB_I_TO_S': SCENARIO_PARAMS['PROB_I_TO_S'], 'VAX_FULL_DAYS': SCENARIO_PARAMS['VAX_FULL_DAYS'], 'VAX_MID_DAYS': SCENARIO_PARAMS['VAX_MID_DAYS'], 'VAX_MID_MULT': SCENARIO_PARAMS['VAX_MID_MULT'], 'VAX_LONG_MULT': SCENARIO_PARAMS['VAX_LONG_MULT'], 'NAT_EFF_MAX': SCENARIO_PARAMS['NAT_EFF_MAX'], 'NAT_EFF_MIN': SCENARIO_PARAMS['NAT_EFF_MIN'], 'NAT_FULL_DAYS': SCENARIO_PARAMS['NAT_FULL_DAYS'], 'NAT_WANE_DAYS': SCENARIO_PARAMS['NAT_WANE_DAYS'], 'External Prevalence': EXTERNAL_PREVALENCE, 'IFR_Structure': 'COVID-like baseline age-specific IFR', 'MC Runs': MC_RUNS_PER_POINT, 'Simulation Days': SIM_DAYS_STRATEGY, 'Initial Infectious Nodes': NUM_SEEDS_PER_RUN})
            if SAVE_DYNAMICS_CURVES:
                for strat in strategies_list:
                    curves = temporal_curves_storage.get(strat, [])
                    if len(curves) == 0:
                        continue
                    curves_matrix = np.vstack(curves)
                    mean_curve = np.mean(curves_matrix, axis=0)
                    sd_curve = np.std(curves_matrix, axis=0)
                    for day, mean_i, sd_i in zip(np.arange(1, len(mean_curve) + 1), mean_curve, sd_curve):
                        dynamics_records.append({'Scenario': SCENARIO_NAME, 'Network_Type': network_type, 'Network_Replicate': replicate_id, 'VES_0': ves0, 'R0': target_r0, 'Coverage': 0.5, 'Strategy': strat, 'Day': int(day), 'Mean Infectious': float(mean_i), 'SD Infectious': float(sd_i)})
    print('[WORKER DONE] Network={}, replicate={}'.format(network_type, replicate_id), flush=True)
    return (pd.DataFrame(temp_results), pd.DataFrame(replicate_records), pd.DataFrame(dynamics_records), diagnostics)

def compute_additional_reduction(results_df, target_coverage=0.5):
    rows = []
    key_cols = ['Network_Type', 'Network_Replicate', 'VES_0', 'R0']
    for keys, group in results_df.groupby(key_cols):
        key_dict = dict(zip(key_cols, keys))
        no_vax = group[np.isclose(group['Coverage'], 0.0, atol=0.01)]
        target = group[np.isclose(group['Coverage'], target_coverage, atol=0.01)]
        random_row = target[target['Strategy'] == 'Random']
        if no_vax.empty or random_row.empty:
            continue
        y0 = no_vax[OUTCOME_COLUMNS].mean(numeric_only=True)
        yr = random_row.iloc[0]
        for _, strat_row in target.iterrows():
            strat = strat_row['Strategy']
            if strat == 'Random':
                continue
            for outcome in OUTCOME_COLUMNS:
                denom = float(y0[outcome])
                if denom == 0 or np.isnan(denom):
                    delta = np.nan
                else:
                    delta = 100.0 * (float(yr[outcome]) - float(strat_row[outcome])) / denom
                rows.append({**key_dict, 'Coverage': target_coverage, 'Strategy': strat, 'Outcome': outcome, 'Y0': denom, 'Y_random': float(yr[outcome]), 'Y_strategy': float(strat_row[outcome]), 'Additional Reduction (%)': delta, 'Positive Point Estimate': float(delta > 0) if not np.isnan(delta) else np.nan})
    return pd.DataFrame(rows)

def aggregate_across_network_replicates(delta_df):
    if delta_df.empty:
        return pd.DataFrame()
    return delta_df.groupby(['Network_Type', 'VES_0', 'R0', 'Strategy', 'Outcome'], as_index=False).agg(Mean_additional_reduction_across_network_replicates=('Additional Reduction (%)', 'mean'), SD_across_network_replicates=('Additional Reduction (%)', 'std'), N_network_replicates=('Network_Replicate', 'nunique'))

def summarize_qualitative_patterns(parameter_level_delta_df):
    if parameter_level_delta_df.empty:
        return pd.DataFrame()
    work = parameter_level_delta_df.copy()
    value_col = 'Mean_additional_reduction_across_network_replicates'
    work['Positive Point Estimate'] = (work[value_col] > 0).astype(float)
    summary = work.groupby(['Network_Type', 'Strategy', 'Outcome'], as_index=False).agg(N_parameter_combinations=(value_col, 'count'), Mean_additional_reduction=(value_col, 'mean'), Median_additional_reduction=(value_col, 'median'), Fraction_positive_point_estimate=('Positive Point Estimate', 'mean'))
    return summary

def plot_network_summary(summary_df, network_types, out_png):
    return summary_df[summary_df['Network_Type'].isin(network_types)].copy()
if __name__ == '__main__':
    print('\n{}'.format('=' * 70), flush=True)
    print('INITIATING NETWORK-STRUCTURE ROBUSTNESS EXECUTION', flush=True)
    print('{}'.format('=' * 70), flush=True)
    print('[INFO] MAX_CORES={}'.format(MAX_CORES), flush=True)
    print('[INFO] R0_LIST={}'.format(R0_LIST), flush=True)
    print('[INFO] VES_0_LIST={}'.format(VES_0_LIST), flush=True)
    print('[INFO] MC_RUNS_PER_POINT={}'.format(MC_RUNS_PER_POINT), flush=True)
    print('[INFO] SAVE_DYNAMICS_CURVES={}'.format(SAVE_DYNAMICS_CURVES), flush=True)
    tasks = []
    for nt in MAIN_TEXT_NETWORK_TYPES:
        if nt == 'Empirical':
            reps = [0]
        else:
            reps = list(range(NETWORK_REPLICATES_MAIN))
        for rep in reps:
            tasks.append((nt, rep))
    for nt in SI_NETWORK_TYPES:
        for rep in range(NETWORK_REPLICATES_SI):
            tasks.append((nt, rep))
    print('[INFO] Total network-replicate tasks: {}'.format(len(tasks)), flush=True)
    print('[INFO] Tasks: {}'.format(tasks), flush=True)
    all_results = []
    all_replicates = []
    all_dynamics = []
    all_diagnostics = []
    failed_tasks = []
    with ProcessPoolExecutor(max_workers=MAX_CORES) as executor:
        futures = {executor.submit(run_network_replicate, nt, rep): (nt, rep) for nt, rep in tasks}
        for future in as_completed(futures):
            nt, rep = futures[future]
            try:
                result_df, replicate_df, dynamics_df, diagnostics = future.result()
                all_results.append(result_df)
                if replicate_df is not None and (not replicate_df.empty):
                    all_replicates.append(replicate_df)
                if dynamics_df is not None and (not dynamics_df.empty):
                    all_dynamics.append(dynamics_df)
                all_diagnostics.append(diagnostics)
            except Exception as e:
                failed_tasks.append((nt, rep, repr(e)))
                print('[ERROR] Network={}, rep={} generated an exception: {}'.format(nt, rep, e), flush=True)
    if failed_tasks:
        raise RuntimeError('One or more network-robustness tasks failed; partial output is not accepted. Failed tasks: {}'.format(failed_tasks))
    if not all_results:
        raise RuntimeError('No network robustness simulation results were generated.')
    final_results_df = pd.concat(all_results, ignore_index=True)
    diagnostics_df = pd.DataFrame(all_diagnostics)
    expected_network_tasks = len(tasks)
    expected_summary_rows = expected_network_tasks * len(VES_0_LIST) * len(R0_LIST) * (1 + len(strategies_list))
    if STRICT_OUTPUT_VALIDATION:
        if len(final_results_df) != expected_summary_rows:
            raise RuntimeError('Incomplete network summary output: expected {} rows, found {}.'.format(expected_summary_rows, len(final_results_df)))
        summary_keys = ['Network_Type', 'Network_Replicate', 'VES_0', 'R0', 'Coverage', 'Strategy']
        if final_results_df.duplicated(summary_keys).any():
            raise RuntimeError('Duplicate network summary settings were detected.')
        if len(diagnostics_df) != expected_network_tasks:
            raise RuntimeError('Expected {} network diagnostic rows, found {}.'.format(expected_network_tasks, len(diagnostics_df)))
    delta_df = compute_additional_reduction(final_results_df, target_coverage=0.5)
    expected_delta_rows = expected_network_tasks * len(VES_0_LIST) * len(R0_LIST) * (len(strategies_list) - 1) * len(OUTCOME_COLUMNS)
    if STRICT_OUTPUT_VALIDATION and len(delta_df) != expected_delta_rows:
        raise RuntimeError('Incomplete network additional-reduction output: expected {} rows, found {}.'.format(expected_delta_rows, len(delta_df)))
    parameter_level_delta_df = aggregate_across_network_replicates(delta_df)
    summary_df = summarize_qualitative_patterns(parameter_level_delta_df)
    main_plot_df = plot_network_summary(summary_df, MAIN_TEXT_NETWORK_TYPES, 'Network_Robustness_Qualitative_Summary_MainText.png')
    all_plot_df = plot_network_summary(summary_df, ALL_NETWORK_TYPES, 'Network_Robustness_Qualitative_Summary_All.png')
    raw_name = 'Network_Robustness_Raw_Results.xlsx'
    final_results_df.to_excel(raw_name, index=False)
    print('[SAVED] {}'.format(raw_name), flush=True)
    if all_replicates:
        replicate_df = pd.concat(all_replicates, ignore_index=True)
        if STRICT_OUTPUT_VALIDATION:
            expected_replicate_rows = expected_summary_rows * MC_RUNS_PER_POINT
            if len(replicate_df) != expected_replicate_rows:
                raise RuntimeError('Incomplete network replicate output: expected {} rows, found {}.'.format(expected_replicate_rows, len(replicate_df)))
            replicate_keys = ['Network_Type', 'Network_Replicate', 'VES_0', 'R0', 'Coverage', 'Strategy', 'MC_Run']
            if replicate_df.duplicated(replicate_keys).any():
                raise RuntimeError('Duplicate network replicate records were detected.')
        replicate_name = 'Network_Robustness_Replicate_Outcomes.csv.gz'
        replicate_df.to_csv(replicate_name, index=False, compression='gzip')
        print('[SAVED] {}'.format(replicate_name), flush=True)
    diag_name = 'Network_Robustness_Network_Diagnostics.xlsx'
    diagnostics_df.to_excel(diag_name, index=False)
    print('[SAVED] {}'.format(diag_name), flush=True)
    run_configuration_df = pd.DataFrame([{'R0_LIST': ','.join(map(str, R0_LIST)), 'VES_0_LIST': ','.join(map(str, VES_0_LIST)), 'Coverage levels': ','.join(map(str, COVERAGE_STEPS.tolist())), 'MC Runs': MC_RUNS_PER_POINT, 'Simulation Days': SIM_DAYS_STRATEGY, 'Initial Infectious Nodes': NUM_SEEDS_PER_RUN, 'Common Random Numbers': USE_COMMON_RANDOM_NUMBERS, 'Strict Output Validation': STRICT_OUTPUT_VALIDATION, 'Allow Duplicate Household Member Keys': ALLOW_DUPLICATE_HOUSEHOLD_MEMBER_KEYS, 'Unique Households': int(df['_household_key'].nunique()), 'Duplicate Household Member Rows': int(duplicated_member_mask.sum()), 'Main Network Replicates': NETWORK_REPLICATES_MAIN, 'SI Network Replicates': NETWORK_REPLICATES_SI, 'External Prevalence': EXTERNAL_PREVALENCE, 'IFR_Structure': 'COVID-like baseline age-specific IFR', **SCENARIO_PARAMS}])
    age_population_df = pd.DataFrame([AGE_GROUP_POPULATION])
    summary_name = 'Network_Robustness_Additional_Reduction_Summary.xlsx'
    with pd.ExcelWriter(summary_name) as writer:
        delta_df.to_excel(writer, sheet_name='Additional_Reduction', index=False)
        parameter_level_delta_df.to_excel(writer, sheet_name='Parameter_Level_Reduction', index=False)
        summary_df.to_excel(writer, sheet_name='Qualitative_Summary', index=False)
        diagnostics_df.to_excel(writer, sheet_name='Network_Diagnostics', index=False)
        main_plot_df.to_excel(writer, sheet_name='Main_Text_Plot_Data', index=False)
        all_plot_df.to_excel(writer, sheet_name='All_Networks_Plot_Data', index=False)
        summary_df[summary_df['Network_Type'].isin(SI_NETWORK_TYPES)].to_excel(writer, sheet_name='SI_Plot_Data', index=False)
        run_configuration_df.to_excel(writer, sheet_name='Run_Configuration', index=False)
        age_population_df.to_excel(writer, sheet_name='Age_Group_Population', index=False)
    print('[SAVED] {}'.format(summary_name), flush=True)
    if SAVE_DYNAMICS_CURVES and all_dynamics:
        dynamics_df = pd.concat(all_dynamics, ignore_index=True)
        dynamics_name = 'Network_Robustness_Dynamics_Curves.xlsx'
        dynamics_df.to_excel(dynamics_name, index=False)
        print('[SAVED] {}'.format(dynamics_name), flush=True)
