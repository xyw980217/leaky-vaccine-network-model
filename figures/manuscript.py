"""Plot Figures 2-5 and S1-S10 from separately supplied data."""
from pathlib import Path
import string

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd

from .common import (HIGH, OLD, STRATEGIES, LABELS, METRICS, TITLES, COLOURS,
                     read_summary, reductions, export_panel, axis_style)


def network_panels(survey, output, formats):
    from ._network_preparation import rebuild_network, STRENGTH
    from ._network_style import panel_network, panel_joint, network_palette
    graph, nodes, _, _ = rebuild_network(survey)
    largest = max(nx.connected_components(graph), key=len)
    positions = nx.spring_layout(graph.subgraph(sorted(largest)), seed=28, weight=None, iterations=300)
    nodes['Shown_In_Panel_A'] = nodes.Node_ID.isin(largest)
    nodes['Display_X'] = nodes.Node_ID.map({k: v[0] for k, v in positions.items()})
    nodes['Display_Y'] = nodes.Node_ID.map({k: v[1] for k, v in positions.items()})
    nodes['Relative_Exposure_Strength'] = nodes[STRENGTH]/nodes[STRENGTH].max()
    number = int(.5*len(nodes))
    high = np.zeros(len(nodes), dtype=bool)
    old = high.copy()
    high[np.argsort(nodes[STRENGTH].to_numpy())[::-1][:number]] = True
    old[np.argsort(nodes.Age.to_numpy())[::-1][:number]] = True
    nodes['Recipient_Group'] = np.select([high & old, high, old],
        ['Both strategies', 'High-exposure only', 'Oldest-first only'], default='Neither strategy')
    edges = pd.DataFrame([{'Source_Node_ID': i, 'Target_Node_ID': j, 'Layer': w['layer']}
                          for i, j, w in graph.edges(data=True)])
    for letter, function in [('A', panel_network), ('B', panel_joint)]:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        fig.subplots_adjust(left=.14, right=.96, top=.85, bottom=.15)
        if letter == 'A':
            function(ax, nodes, edges)
            colourbar = (ScalarMappable(norm=Normalize(0, 1), cmap=network_palette()),
                         'Relative exposure strength', [0, .5, 1])
        else:
            function(ax, nodes)
            colourbar = None
        export_panel(fig, output, 'Fig2_'+letter, formats, colourbar)
    print(f'Network: {len(nodes)} residents; {graph.number_of_edges()} edges; '
          f'largest component {len(largest)} residents; shared recipients {(high & old).sum()}.')


def heatmaps(data, strategy, prefix, output, formats):
    delta = reductions(data)
    at_half = delta[np.isclose(delta.Coverage, .5)]
    selected = at_half[at_half.Strategy == strategy]
    for letter, metric, title in zip('ABC', METRICS, TITLES):
        table = selected.pivot(index='VES_0', columns='R0', values=metric+'_extra').sort_index(ascending=False)
        bound = max(.5, at_half[metric+'_extra'].abs().max())
        fig, ax = plt.subplots(figsize=(13, 4.2))
        fig.subplots_adjust(left=.09, right=.89, bottom=.25, top=.85)
        im = ax.imshow(table, cmap='RdYlGn', vmin=-bound, vmax=bound, aspect='auto')
        for (i, j), value in np.ndenumerate(table.to_numpy()):
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(value) > .58*bound else '#303030')
        ax.set_xticks(range(len(table.columns)), [f'{r:g}' for r in table.columns], rotation=45)
        ax.set_yticks(range(len(table.index)), [f'{v:.2f}' for v in table.index])
        ax.set_xlabel(r'Basic reproduction number ($R_0$)')
        ax.set_ylabel(r'Baseline vaccine efficacy ($VE_{S,0}$)')
        ax.set_xticks(np.arange(-.5, len(table.columns)), minor=True)
        ax.set_yticks(np.arange(-.5, len(table.index)), minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
        ax.tick_params(which='minor', length=0)
        axis_style(ax, title, letter)
        fig.colorbar(im, ax=ax, fraction=.025, pad=.035, label='Additional reduction (pp)')
        export_panel(fig, output, prefix+'_'+letter, formats, (im, 'Additional reduction (pp)', None))


def endpoint_map(data, output, formats, all_or_nothing=False):
    half = data[np.isclose(data.Coverage, .5)]
    high = half[half.Strategy == HIGH].set_index(['VES_0', 'R0'])
    old = half[half.Strategy == OLD].set_index(['VES_0', 'R0'])
    difference = high[['Infection Rate', 'Peak Size']] - old[['Infection Rate', 'Peak Size']]
    fip, peak = difference['Infection Rate'], difference['Peak Size']
    codes = np.select([(fip < 0) & (peak < 0), (fip > 0) & (peak < 0),
                       (fip > 0) & (peak > 0)], [0, 1, 2], default=3)
    labels = ['High-exposure advantage', 'Opposite endpoint rankings' if all_or_nothing else 'Trade-off',
              'Oldest-first advantage', 'Other endpoint ordering or tie']
    colours = ['#AAD8C5', '#C6B4DE' if all_or_nothing else '#A9C7EF', '#EBAFC3', '#DDDDDD']
    table = pd.Series(codes, index=difference.index).unstack('R0').sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.subplots_adjust(left=.1, right=.97, top=.86, bottom=.33)
    ax.imshow(table, aspect='auto', cmap=ListedColormap(colours), vmin=-.5, vmax=3.5)
    ax.set_xticks(range(len(table.columns)), [f'{r:g}' for r in table.columns])
    ax.set_yticks(range(len(table.index)), [f'{v:.2f}' for v in table.index])
    ax.set_xticks(np.arange(-.5, len(table.columns)), minor=True)
    ax.set_yticks(np.arange(-.5, len(table.index)), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    ax.set_xlabel(r'Basic reproduction number ($R_0$)')
    ax.set_ylabel(r'Baseline vaccine efficacy ($VE_{S,0}$)')
    axis_style(ax, 'Endpoint-defined transmission patterns', 'A' if not all_or_nothing else '')
    used = sorted(np.unique(codes))
    ax.legend(handles=[Patch(color=colours[i], label=labels[i]) for i in used],
              loc='upper center', bbox_to_anchor=(.5, -.36), ncol=min(3, len(used)), frameon=False)
    export_panel(fig, output, 'FigS3' if all_or_nothing else 'Fig5_A', formats)
    print('Endpoint classification:', dict(zip([labels[i] for i in used], [(codes == i).sum() for i in used])))


def hybrid_panels(folder, output, formats):
    folder = Path(folder)
    summary = pd.read_csv(folder/'Quota_Mixture_Tradeoff_AllPoint_Summary.csv')
    summary = summary[np.isclose(summary.VES_0, .8)]
    cmap = plt.get_cmap('viridis')
    for r0 in [1, 3, 18.6]:
        data = summary[np.isclose(summary.R0, r0)].sort_values('Alpha High Exposure')
        if data.empty:
            raise ValueError(f'Missing hybrid trajectory for R0={r0}.')
        x, y = 100*data['Final Infection Proportion Mean'], data['Peak Size Mean']
        fig, ax = plt.subplots(figsize=(6.5, 5))
        fig.subplots_adjust(left=.17, right=.87, bottom=.18, top=.87)
        ax.plot(x, y, color='#AAB1B8', lw=1, zorder=1)
        im = ax.scatter(x, y, c=data['Alpha High Exposure'], cmap=cmap, vmin=0, vmax=1, s=24, zorder=2)
        for alpha, marker, label in [(0, 's', 'Oldest-first'), (.5, 'D', 'Hybrid'), (1, '^', 'High-exposure')]:
            select = np.isclose(data['Alpha High Exposure'], alpha)
            ax.scatter(x[select], y[select], marker=marker, s=90, color=cmap(float(alpha)),
                       edgecolors='black', linewidths=.7, label=label, zorder=4)
        ax.set_xlabel('Final infection proportion (%)')
        ax.set_ylabel('Epidemic peak size (individuals)')
        axis_style(ax, rf'$R_0={r0:g}$', 'B')
        ax.legend(frameon=False, fontsize=9)
        fig.colorbar(im, ax=ax, label=r'$\alpha$', fraction=.05, pad=.06)
        export_panel(fig, output, f'Fig5_B_R0_{r0:g}', formats, (im, r'$\alpha$', [0, .5, 1]))
    for letter, metric, title, unit in [('C', 'FIP', 'Final infection proportion', 'pp'),
                                        ('D', 'Peak', 'Epidemic peak size', 'individuals')]:
        data = pd.read_csv(folder/'fig5_CD_endpoint_decomposition'/f'Panel_{letter}_{metric}_Plot_Ready.csv').set_index('R0')
        bound = max(abs(data.to_numpy().min()), abs(data.to_numpy().max()), .01)
        for r0 in [1, 3, 18.6]:
            row = data.loc[r0]
            fig, ax = plt.subplots(figsize=(10, 3.25))
            fig.subplots_adjust(left=.09, right=.89, bottom=.32, top=.76)
            cmap = LinearSegmentedColormap.from_list('strategy_difference', ['#0072B2', '#FFFFFF', '#D55E00'])
            im = ax.imshow([row.to_numpy()], cmap=cmap, vmin=-bound, vmax=bound, aspect='auto')
            for j, value in enumerate(row):
                ax.text(j, 0, f'{value:.2f}', ha='center', va='center', fontsize=10,
                        color='white' if abs(value) > .62*bound else '#202020')
            labels = list(data.columns[:-1])+['Total']
            ax.set_xticks(range(len(row)), labels, rotation=40, ha='right')
            ax.set_yticks([])
            ax.set_xlabel('Exposure strength percentile (low to high)')
            ax.axvline(4.5, color='black', ls='--', lw=1)
            axis_style(ax, rf'{title} ({unit}), $R_0={r0:g}$', letter)
            fig.colorbar(im, ax=ax, fraction=.025, pad=.035, label=f'Difference ({unit})')
            export_panel(fig, output, f'Fig5_{letter}_R0_{r0:g}', formats, (im, f'Difference ({unit})', None))


def paired_changes(folder, model):
    data = pd.read_csv(Path(folder)/'Quota_Mixture_Tradeoff_AllReplicate_Outcomes.csv.gz')
    data = data[np.isclose(data.VES_0, .8) & np.isclose(data.Coverage, .5)]
    alpha = 'Alpha High Exposure'
    keys = ['R0', 'MC_Run']
    if data.duplicated(keys+[alpha]).any():
        raise ValueError('Replicate pairing is ambiguous.')
    baseline = data[np.isclose(data[alpha], 0)][keys+['Final Infection Proportion', 'Peak Size']]
    paired = data.merge(baseline, on=keys, suffixes=('', '_zero'), validate='many_to_one')
    rows = []
    for (r0, a), group in paired.groupby(['R0', alpha]):
        for metric, scale in [('Final Infection Proportion', 100), ('Peak Size', 1)]:
            values = scale*(group[metric]-group[metric+'_zero'])
            mean, se = values.mean(), values.std(ddof=1)/np.sqrt(len(values))
            rows.append(dict(Model=model, R0=r0, Alpha=a, Metric=metric, Mean=mean,
                             Lower=mean-1.96*se, Upper=mean+1.96*se))
    return pd.DataFrame(rows)


def comparison(leaky_folder, aon_folder, output, formats):
    values = pd.concat([paired_changes(leaky_folder, 'Leaky'),
                        paired_changes(aon_folder, 'All-or-nothing')], ignore_index=True)
    # Match each outcome scale between protection models within each R0 column.
    for column, r0 in enumerate([1, 7, 18.6]):
        subsets = values[np.isclose(values.R0, r0)]
        spans = []
        for metric in ['Final Infection Proportion', 'Peak Size']:
            current = subsets[subsets.Metric == metric]
            lo, hi = min(0, current.Lower.min()), max(0, current.Upper.max())
            spans.append((lo, hi))
        negative_fraction = max((-lo/(hi-lo) if hi > lo else .5 for lo, hi in spans))
        negative_fraction = np.clip(negative_fraction, .15, .85)
        limits = []
        for lo, hi in spans:
            extent = max(-lo/negative_fraction, hi/(1-negative_fraction), .001)*1.08
            limits.append((-extent*negative_fraction, extent*(1-negative_fraction)))
        for row, model in enumerate(['Leaky', 'All-or-nothing']):
            fig, left = plt.subplots(figsize=(6.5, 4.7))
            fig.subplots_adjust(left=.19, right=.8, bottom=.19, top=.84)
            right = left.twinx()
            for ax, metric, colour, marker, ls, ylim in zip([left, right], ['Final Infection Proportion', 'Peak Size'],
                    ['#0072B2', '#D55E00'], ['o', 's'], ['-', '--'], limits):
                series = subsets[(subsets.Model == model) & (subsets.Metric == metric)].sort_values('Alpha')
                if series.empty:
                    raise ValueError(f'Missing comparison: {model}, R0={r0}.')
                ax.fill_between(series.Alpha, series.Lower, series.Upper, color=colour, alpha=.14, lw=0)
                ax.plot(series.Alpha, series.Mean, color=colour, marker=marker, markevery=5,
                        markersize=4, ls=ls, label=metric)
                ax.set_ylim(*ylim)
                ax.tick_params(axis='y', colors=colour)
                ax.spines['top'].set_visible(False)
            left.axhline(0, color='#777777', ls='--', lw=1)
            left.set_xlabel(r'High-exposure quota fraction $\alpha$')
            left.set_ylabel('Change in final infection proportion (pp)', color='#0072B2')
            right.set_ylabel('Change in epidemic peak size (individuals)', color='#D55E00')
            axis_style(left, rf'{model}, $R_0={r0:g}$', string.ascii_uppercase[3*row+column])
            handles = left.get_lines()[:1]+right.get_lines()[:1]
            left.legend(handles=handles, loc='best', fontsize=8, frameon=False)
            export_panel(fig, output, 'FigS4_'+string.ascii_uppercase[3*row+column], formats)


def coverage_panels(data, output, formats):
    delta = reductions(data)
    for row, strategy in enumerate(STRATEGIES):
        selected = delta[delta.Strategy == strategy]
        for column, (metric, title) in enumerate(zip(METRICS, TITLES)):
            group = selected.groupby('Coverage')[metric+'_extra']
            mean, lower, upper = group.mean(), group.quantile(.25), group.quantile(.75)
            fig, ax = plt.subplots(figsize=(6.2, 4.7))
            fig.subplots_adjust(left=.18, right=.96, bottom=.18, top=.86)
            ax.fill_between(mean.index*100, lower, upper, alpha=.17, color=COLOURS[strategy], label='Interquartile range')
            ax.plot(mean.index*100, mean, 'o-', color=COLOURS[strategy], markersize=4, label='Mean across settings')
            ax.axhline(0, color='#777777', ls='--', lw=1)
            ax.axvline(50, color='#777777', ls=':', lw=1)
            ax.set(xlabel='Vaccination coverage (%)', ylabel='Additional reduction (pp)')
            letter = string.ascii_uppercase[3*row+column]
            axis_style(ax, LABELS[strategy]+'\n'+title, letter)
            ax.legend(frameon=False, fontsize=9)
            export_panel(fig, output, 'FigS5_'+letter, formats)


def disease_panels(path, output, formats, strategy):
    path = Path(path)
    data = pd.read_csv(path/'figure_source_data.csv' if path.is_dir() else path)
    names = {'Baseline': 'COVID-like', 'Disease_A': 'Influenza-like', 'Disease_B': 'RSV-like'}
    metrics = ['Final Infection Proportion', 'Epidemic Peak Size', 'Expected Deaths']
    selected = data[data.Strategy == strategy]
    for index, (metric, title) in enumerate(zip(metrics, TITLES)):
        part = selected[selected.Outcome == metric]
        table = part.pivot(index='Scenario', columns='VES_0', values='Additional_Reduction_Percent')
        table = table.reindex(index=list(names)).sort_index(axis=1)
        bound = max(.5, data.loc[data.Outcome == metric, 'Additional_Reduction_Percent'].abs().max())
        fig, ax = plt.subplots(figsize=(7, 4.8))
        fig.subplots_adjust(left=.24, right=.86, bottom=.2, top=.85)
        im = ax.imshow(table, cmap='RdYlGn', aspect='auto', vmin=-bound, vmax=bound)
        for (i, j), value in np.ndenumerate(table.to_numpy()):
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color='white' if abs(value) > .6*bound else '#202020')
        labels = []
        for scenario in table.index:
            r0 = part.loc[part.Scenario == scenario, 'R0'].unique()
            if len(r0) != 1:
                raise ValueError('Disease panels require one reference R0 per scenario.')
            labels.append(names[scenario]+f'\n'+rf'$R_0={r0[0]:g}$')
        ax.set_yticks(range(3), labels)
        ax.set_xticks(range(len(table.columns)), [f'{v:.2f}' for v in table.columns])
        ax.set_xlabel(r'Baseline vaccine efficacy ($VE_{S,0}$)')
        ax.set_xticks(np.arange(-.5, len(table.columns)), minor=True)
        ax.set_yticks(np.arange(-.5, len(table.index)), minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
        ax.tick_params(which='minor', length=0)
        axis_style(ax, title, string.ascii_uppercase[index])
        fig.colorbar(im, ax=ax, fraction=.05, pad=.04, label='Additional reduction (pp)')
        prefix = 'FigS6' if strategy == HIGH else 'FigS7'
        export_panel(fig, output, prefix+'_'+string.ascii_uppercase[index], formats,
                     (im, 'Additional reduction (pp)', None))


def network_sensitivity(path, output, formats):
    path = Path(path)
    if path.is_dir():
        path = path/'Network_Robustness_Additional_Reduction_Summary.xlsx'
    if path.suffix == '.xlsx':
        data = pd.read_excel(path, sheet_name='Parameter_Level_Reduction').rename(
            columns={'Mean_additional_reduction_across_network_replicates': 'Additional_reduction'})
    else:
        data = pd.read_csv(path)
    kinds = ['Empirical', 'CommunityResampled', 'SmallWorldCommunity', 'HouseholdAttenuatedCompensated',
             'ERRandom', 'BAScaleFree', 'CompletelyRandomized']
    labels = ['Reference', 'Community\nresampled', 'Small-world\ncommunity', 'Household\nattenuated',
              'ER random', 'BA scale-free', 'Fully\nrandomised']
    metrics = ['Final Infection Proportion', 'Epidemic Peak Size', 'Expected Deaths']
    rng = np.random.RandomState(12)
    for row, strategy in enumerate(STRATEGIES):
        for column, (metric, title) in enumerate(zip(metrics, TITLES)):
            part = data[(data.Strategy == strategy) & (data.Outcome == metric)]
            arrays = [part.loc[part.Network_Type == k, 'Additional_reduction'].to_numpy() for k in kinds]
            if any(len(a) == 0 for a in arrays):
                raise ValueError('Missing network type in parameter-level results.')
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.subplots_adjust(left=.12, right=.97, bottom=.27, top=.84)
            ax.boxplot(arrays, positions=np.arange(7), widths=.52, showfliers=False,
                       showmeans=True, meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black'))
            for index, array in enumerate(arrays):
                ax.scatter(index+rng.uniform(-.16, .16, len(array)), array, s=18,
                           color=COLOURS[strategy], alpha=.65, zorder=3)
            ax.axhline(0, color='#777777', ls='--', lw=1)
            ax.axvline(3.5, color='#777777', ls=':', lw=1)
            ax.set_xticks(range(7), labels, fontsize=9)
            ax.set_ylabel('Additional reduction (pp)')
            letter = string.ascii_uppercase[row*3+column]
            axis_style(ax, LABELS[strategy]+'\n'+title, letter)
            ax.legend(handles=[Line2D([], [], marker='D', color='none', markeredgecolor='black',
                      markerfacecolor='white', label='Mean')], frameon=False)
            export_panel(fig, output, 'FigS8_'+letter, formats)


def uncertainty_panels(data, output, formats):
    half = data[np.isclose(data.Coverage, .5)]
    for index, (metric, title) in enumerate(zip(METRICS, TITLES)):
        scale = 100 if metric == 'Infection Rate' else 1
        arrays = [1.96*half.loc[half.Strategy == s, metric+' SE'].to_numpy()*scale
                  for s in ['Random', HIGH, OLD]]
        fig, ax = plt.subplots(figsize=(6.3, 4.8))
        fig.subplots_adjust(left=.18, right=.96, bottom=.19, top=.85)
        ax.boxplot(arrays, tick_labels=['Random', 'High-exposure', 'Oldest-first'], showfliers=False,
                   showmeans=True, meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black'))
        rng = np.random.RandomState(25)
        for position, (array, strategy) in enumerate(zip(arrays, ['Random', HIGH, OLD]), start=1):
            ax.scatter(position+rng.uniform(-.16, .16, len(array)), array, s=13,
                       color=COLOURS[strategy], alpha=.55)
        ax.legend(handles=[Line2D([], [], marker='D', color='none', markeredgecolor='black',
                  markerfacecolor='white', label='Mean')], frameon=False)
        ax.set_ylabel('95% interval half-width ('+('pp' if index == 0 else 'individuals' if index == 1 else 'expected deaths')+')')
        axis_style(ax, title, string.ascii_uppercase[index])
        export_panel(fig, output, 'FigS9_'+string.ascii_uppercase[index], formats)
    delta = reductions(data)
    delta = delta[np.isclose(delta.Coverage, .5)]
    category_colours = ['#29916B', '#A9AFB5', '#CD5C5C']
    for index, strategy in enumerate(STRATEGIES):
        group = delta[delta.Strategy == strategy]
        counts = []
        for metric in METRICS:
            value, halfwidth = group[metric+'_extra'], 1.96*group[metric+'_extra_se']
            counts.append([(value-halfwidth > 0).sum(), ((value-halfwidth <= 0) & (value+halfwidth >= 0)).sum(),
                           (value+halfwidth < 0).sum()])
        counts = np.asarray(counts)
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        fig.subplots_adjust(left=.13, right=.97, bottom=.26, top=.84)
        bottom = np.zeros(3)
        for k, label in enumerate(['Above zero', 'Includes zero', 'Below zero']):
            ax.bar(np.arange(3), counts[:, k], bottom=bottom, color=category_colours[k], label=label)
            for j, value in enumerate(counts[:, k]):
                if value:
                    ax.text(j, bottom[j]+value/2, str(value), ha='center', va='center', color='white')
            bottom += counts[:, k]
        ax.set_xticks(range(3), ['Final infection\nproportion', 'Epidemic peak\nsize', 'Expected deaths'])
        ax.set_ylabel('Number of parameter settings')
        axis_style(ax, LABELS[strategy], 'DE'[index])
        ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.2))
        export_panel(fig, output, 'FigS9_'+'DE'[index], formats)


def representative_intervals(data, output, formats):
    delta = reductions(data)
    delta = delta[np.isclose(delta.Coverage, .5)]
    settings = [(HIGH, 'Peak Size', 2, .5), (HIGH, 'Infection Rate', 4, .8),
                (OLD, 'Expected Deaths', 1.2, .8), (OLD, 'Peak Size', 1.5, .8),
                (HIGH, 'Peak Size', 18.6, .5)]
    values, errors, labels = [], [], []
    for strategy, metric, r0, ve in settings:
        chosen = delta[(delta.Strategy == strategy) & np.isclose(delta.R0, r0) & np.isclose(delta.VES_0, ve)]
        if len(chosen) != 1:
            raise ValueError(f'Missing representative interval: {strategy}, {r0}, {ve}.')
        values.append(chosen.iloc[0][metric+'_extra'])
        errors.append(1.96*chosen.iloc[0][metric+'_extra_se'])
        labels.append(LABELS[strategy]+', '+TITLES[METRICS.index(metric)]+'\n'+rf'$R_0={r0:g},\ VE_{{S,0}}={ve:g}$')
    fig, ax = plt.subplots(figsize=(13, 5.2))
    fig.subplots_adjust(left=.09, right=.97, bottom=.31, top=.87)
    for i, (value, error, setting) in enumerate(zip(values, errors, settings)):
        ax.errorbar(i, value, yerr=error, fmt='o', color=COLOURS[setting[0]], capsize=4)
    labels = [label.replace(', Final infection proportion', '\nFinal infection proportion')
                    .replace(', Epidemic peak size', '\nEpidemic peak size')
                    .replace(', Expected deaths', '\nExpected deaths') for label in labels]
    ax.axhline(0, color='#777777', ls='--', lw=1)
    ax.set_xticks(range(5), labels, fontsize=9)
    ax.set_ylabel('Additional reduction (pp)')
    axis_style(ax, 'Representative strategy differences')
    export_panel(fig, output, 'FigS10', formats)
