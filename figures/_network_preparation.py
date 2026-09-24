"""Reconstruct the survey-derived network for Figure 2."""
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
AGE_VALUES = [15.0, 24.0, 34.5, 44.5, 54.5, 64.5, 75.0]
STRENGTH = 'Total_Weighted_Exposure_Strength'
FREQ_MAP = {'\u65e0': 0, '<1\u6b21': 0.5, '1-2\u6b21': 1.5, '3-4\u6b21': 3.5, '\u22655\u6b21': 5.5, 'a': 5.5, 'b': 3.5, 'c': 1.5, 'd': 0.5}
DUR_MAP = {'\u65e0': 0, '2\u5c0f\u65f6\u4ee5\u5185': 1, '2-5\u5c0f\u65f6': 3.5, '5\u5c0f\u65f6\u4ee5\u4e0a': 6, 'a': 1, 'b': 3.5, 'c': 6}

def normalize(s):
    return str(s).replace('\ufeff', '').strip().replace(' ', '').replace('\uff08', '(').replace('\uff09', ')').replace('/', '')

def pick_column(df, candidates):
    names = {c: normalize(c) for c in df.columns}
    wanted = [normalize(c) for c in candidates]
    for c, nc in names.items():
        if nc in wanted:
            return c
    for c, nc in names.items():
        if any((w in nc for w in wanted)) or any((nc in w for w in wanted)):
            return c
    raise KeyError(f'Missing column: {candidates}')

def rebuild_network(survey):
    raw = pd.read_excel(survey, sheet_name=0)
    n = len(raw)
    ccommunity = pick_column(raw, ['\u5c0f\u533a\u540d\u79f0', '\u5c0f\u533a', '\u793e\u533a\u540d\u79f0', '\u793e\u533a'])
    caddress = pick_column(raw, ['\u4f4f\u6237\u8be6\u7ec6\u5730\u5740', '\u4f4f\u5740'])
    cwork = pick_column(raw, ['\u5de5\u4f5c\u72b6\u6001'])
    ccontacts = pick_column(raw, ['\u4eba\u5458\u6570\u91cf', '\u5b66\u4e60\u573a\u6240\u7684\u4eba\u5458\u6570\u91cf'])
    cage = pick_column(raw, ['\u5e74\u9f84'])
    household = raw[ccommunity].astype(str).str.strip() + '|' + raw[caddress].astype(str).str.strip()
    hh_ids, _ = pd.factorize(household)
    age = raw[cage].astype(str).str.strip().map(dict(zip('abcdefg', AGE_VALUES))).fillna(45.0).to_numpy()
    steps = pd.DataFrame({'Node_ID': np.arange(n), 'Source_Excel_Row': np.arange(n) + 2, 'Anonymous_Household_ID': hh_ids, 'Household_Size': pd.Series(hh_ids).map(pd.Series(hh_ids).value_counts()), 'Raw_Age_Code': raw[cage].astype(str), 'Raw_Work_Status': raw[cwork].astype(str), 'Raw_Fixed_Contact_Code': raw[ccontacts].astype(str), 'Mapped_Age': age})
    nonfixed = np.zeros(n)
    for k in range(1, 6):
        if f'\u9891\u7387{k}' in raw.columns:
            freq = raw[f'\u9891\u7387{k}'].map(lambda x: FREQ_MAP.get(str(x).strip(), 5.5 if '5\u6b21' in str(x).strip() else 0.5)).fillna(0).to_numpy()
            dur = raw[f'\u65f6\u957f{k}'].map(lambda x: DUR_MAP.get(str(x).strip(), 3.5 if '2-5' in str(x).strip() else 0.0)).fillna(0).to_numpy()
            nonfixed += freq * dur / 7.0
            steps[f'Raw_Frequency_{k}'] = raw[f'\u9891\u7387{k}'].astype(str)
            steps[f'Raw_Duration_{k}'] = raw[f'\u65f6\u957f{k}'].astype(str)
            steps[f'Mapped_Frequency_Per_Week_{k}'] = freq
            steps[f'Mapped_Duration_Hours_{k}'] = dur
            steps[f'Daily_Nonfixed_Hours_{k}'] = freq * dur / 7.0
    work = np.array([str(s).strip() in ['b', 'B', 'd', 'D', 'a', 'A'] or any((t in str(s).strip() for t in ['\u5de5', '\u73ed', '\u5b66'])) for s in raw[cwork]], dtype=bool)
    fixed = work.astype(float) * 8.0

    def contact_count(x):
        s = str(x).lower().strip()
        for k, v in {'a': 3.0, 'b': 10.0, 'c': 20.0, 'd': 30.0}.items():
            if k in s:
                return v
        return 0.0
    virtual_contacts = raw[ccontacts].map(contact_count).fillna(0).to_numpy()
    g = nx.Graph()
    g.add_nodes_from(range(n))
    members = defaultdict(list)
    for i, hh in enumerate(household):
        members[hh].append(i)
    for nodes in members.values():
        if len(nodes) <= 1:
            continue
        w = 12.0 / (len(nodes) - 1)
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                g.add_edge(nodes[a], nodes[b], weight=w, layer='home')
    rng = np.random.RandomState(42)
    ids = np.arange(n)
    for i in range(n):
        internal_time = nonfixed[i] * 0.3
        if internal_time > 0:
            encounters = int(round(internal_time / 0.25))
            if encounters > 0:
                targets = rng.choice(ids[ids != i], size=min(encounters, n - 1), replace=False)
                for j in targets:
                    if g.has_edge(i, j):
                        g[i][j]['weight'] += 0.25
                    else:
                        g.add_edge(i, j, weight=0.25, layer='community_random')
    internal = np.array([g.degree(i, weight='weight') for i in range(n)])
    external = virtual_contacts * fixed + nonfixed * (1.0 - 0.3) * 4.0
    total = internal + external
    neighbor = np.array([np.mean([total[j] for j in g.neighbors(i)]) if g.degree(i) else 0.0 for i in range(n)])
    attrs = pd.DataFrame({'Node_ID': ids, 'Anonymous_Household_ID': hh_ids, 'Age': age, 'Work_School_Activity_Status': work.astype(int), 'Daily_Activity_Duration': np.clip(fixed + nonfixed, 0, 12.0), 'Internal_Weighted_Strength': internal, 'External_Virtual_Strength': external, STRENGTH: total, 'Average_Neighbour_Exposure_Strength': neighbor})
    steps['Work_School_Activity_Status'] = work.astype(int)
    steps['Fixed_Activity_Hours'] = fixed
    steps['Nonfixed_Activity_Hours'] = nonfixed
    steps['Daily_Activity_Hours_Before_Clipping'] = fixed + nonfixed
    steps['Daily_Activity_Duration'] = attrs['Daily_Activity_Duration']
    steps['Mapped_Fixed_Contact_Count'] = virtual_contacts
    steps['Internal_Nonfixed_Hours'] = nonfixed * 0.3
    steps['Requested_Community_Encounters'] = [int(round(v * 0.3 / 0.25)) if v * 0.3 > 0 else 0 for v in nonfixed]
    steps['External_Fixed_Contribution'] = virtual_contacts * fixed
    steps['External_Nonfixed_Contribution'] = nonfixed * (1.0 - 0.3) * 4.0
    for col in ['Internal_Weighted_Strength', 'External_Virtual_Strength', STRENGTH]:
        steps[col] = attrs[col]
    mapping_info = {'survey_sheet_index': 0, 'row_order_preserved': True, 'detected_columns': {'community': ccommunity, 'address': caddress, 'work': cwork, 'fixed_contact_count': ccontacts, 'age': cage}, 'age_code_to_model_value': dict(zip('abcdefg', AGE_VALUES)), 'frequency_mapping': FREQ_MAP, 'duration_mapping': DUR_MAP, 'unmapped_age_rows_using_original_default_45': int((~raw[cage].astype(str).str.strip().isin(list('abcdefg'))).sum()), 'address_strings_exported': False}
    return (g, attrs, steps, mapping_info)
