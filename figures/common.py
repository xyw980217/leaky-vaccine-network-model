"""Shared data calculations and individual-panel export."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.cm import ScalarMappable
from matplotlib.text import Text
import numpy as np
import pandas as pd

HIGH = 'High Strength (Node)'
OLD = 'Priority: Oldest First'
STRATEGIES = [HIGH, OLD]
LABELS = {HIGH: 'High-exposure', OLD: 'Oldest-first', 'Random': 'Random'}
METRICS = ['Infection Rate', 'Peak Size', 'Expected Deaths']
TITLES = ['Final infection proportion', 'Epidemic peak size', 'Expected deaths']
COLOURS = {HIGH: '#0072B2', OLD: '#D55E00', 'Random': '#888888'}
SUMMARY = 'Simulation_Results_All_Scenarios_With_Uncertainty.csv.gz'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})


def read_summary(path):
    path = Path(path)
    if path.is_dir():
        path = path / SUMMARY
    data = pd.read_csv(path)
    data['Coverage'] = data['Coverage'].round(2)
    return data[data.Strategy.isin(['Random', HIGH, OLD])].copy()


def reductions(data):
    keys = ['VES_0', 'R0']
    zero = data[(data.Coverage == 0) & (data.Strategy == 'Random')]
    random = data[(data.Coverage > 0) & (data.Strategy == 'Random')]
    target = data[(data.Coverage > 0) & data.Strategy.isin(STRATEGIES)]
    if zero.duplicated(keys).any() or random.duplicated(keys+['Coverage']).any():
        raise ValueError('Duplicate parameter settings in summary data.')
    joined = target.merge(random, on=keys+['Coverage'], suffixes=('', '_random'), validate='many_to_one')
    joined = joined.merge(zero[keys+METRICS].rename(columns={m: m+'_zero' for m in METRICS}),
                          on=keys, validate='many_to_one')
    for metric in METRICS:
        denominator = joined[metric+'_zero']
        if (denominator <= 0).any():
            raise ValueError('A positive no-vaccination mean is required.')
        joined[metric+'_extra'] = 100*(joined[metric+'_random']-joined[metric])/denominator
        if metric+' SE' in joined:
            joined[metric+'_extra_se'] = 100*np.hypot(joined[metric+' SE'], joined[metric+' SE_random'])/denominator
    return joined


def axis_style(ax, title='', letter=''):
    ax.set_title(title, loc='left', pad=12)
    if letter:
        ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontsize=16, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out')


def save_formats(fig, folder, stem, formats):
    folder.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(folder/f'{stem}.{extension}', dpi=300, facecolor='white')


def export_panel(fig, destination, name, formats=('png', 'pdf'), colourbar=None):
    destination = Path(destination)
    save_formats(fig, destination/'with_text', name, formats)
    handles, labels = [], []
    for legend in fig.findobj(Legend):
        handles.extend(legend.legend_handles)
        labels.extend(text.get_text() for text in legend.get_texts())
    if handles:
        pairs = dict(zip(labels, handles))
        legend_fig = plt.figure(figsize=(max(5, min(13, len(pairs)*2.6)), 1.25))
        legend_fig.legend(list(pairs.values()), list(pairs), loc='center', frameon=False,
                          ncol=min(3, len(pairs)))
        save_formats(legend_fig, destination/'legends', name+'_legend', formats)
        plt.close(legend_fig)
    if colourbar is not None:
        mappable, label, ticks = colourbar
        legend_fig, cax = plt.subplots(figsize=(5, 1.1))
        legend_fig.subplots_adjust(left=.12, right=.93, bottom=.55, top=.83)
        separate_mappable = ScalarMappable(norm=mappable.norm, cmap=mappable.cmap)
        cb = legend_fig.colorbar(separate_mappable, cax=cax, orientation='horizontal', ticks=ticks)
        cb.set_label(label)
        save_formats(legend_fig, destination/'legends', name+'_colourbar', formats)
        plt.close(legend_fig)
    for legend in fig.findobj(Legend):
        legend.set_visible(False)
    for ax in fig.axes:
        if hasattr(ax, '_colorbar'):
            ax.set_visible(False)
    for ax in getattr(fig, '_network_colourbar_axes', []):
        ax.set_visible(False)
    for text in fig.findobj(Text):
        text.set_visible(False)
    save_formats(fig, destination/'without_text', name, formats)
    plt.close(fig)
