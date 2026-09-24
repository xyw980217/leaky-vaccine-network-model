"""Draw the network and vaccination-target panels in Figure 2."""
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
RAW = 'Total_Weighted_Exposure_Strength'
REL = 'Relative_Exposure_Strength'
AGE_VALUES = [15.0, 24.0, 34.5, 44.5, 54.5, 64.5, 75.0]
AGE_LABELS = ['\u226418', '19\u201329', '30\u201339', '40\u201349', '50\u201359', '60\u201369', '\u226570']
HIGH, OLD, SHARED, NONE = ('#0072B2', '#D55E00', '#8B61A8', '#BFC4C9')

def style(ax, title, label):
    ax.set_title(title, loc='left', fontsize=13, pad=14, fontweight='medium')
    ax.text(-0.105, 1.06, label, transform=ax.transAxes, fontsize=18, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E6E9EC', linewidth=0.7)

def network_palette():
    return LinearSegmentedColormap.from_list('Blues_visible_on_white', matplotlib.colormaps['Blues'](np.linspace(0.28, 0.98, 256)))

def panel_network(ax, nodes, edges):
    style(ax, 'Largest connected component', 'A')
    main = nodes.loc[nodes.Shown_In_Panel_A].set_index('Node_ID')
    positions = main[['Display_X', 'Display_Y']]
    main_edges = edges.loc[edges.Source_Node_ID.isin(main.index) & edges.Target_Node_ID.isin(main.index)]
    for layer, color, alpha, width in [('community_random', '#7F8992', 0.52, 0.5), ('home', '#3B4650', 0.78, 0.75)]:
        layer_edges = main_edges.loc[main_edges.Layer == layer]
        segments = np.stack([positions.loc[layer_edges.Source_Node_ID].to_numpy(), positions.loc[layer_edges.Target_Node_ID].to_numpy()], axis=1)
        ax.add_collection(LineCollection(segments, colors=color, linewidths=width, alpha=alpha, zorder=1))
    order = main.sort_values(REL, kind='stable')
    sc = ax.scatter(order.Display_X, order.Display_Y, c=order[REL], cmap=network_palette(), vmin=0, vmax=1, s=5.5, linewidths=0.08, edgecolors='white', zorder=3)
    ax.margins(0.045)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_axis_off()
    ax.legend(handles=[Line2D([], [], color='#3B4650', alpha=0.78, lw=1.5, label='Household'), Line2D([], [], color='#7F8992', alpha=0.52, lw=1.5, label='Community')], loc='upper left', frameon=False, fontsize=9)
    ax.text(0.01, 0.035, f'{len(main):,} nodes ({100 * len(main) / len(nodes):.1f}% of residents)', transform=ax.transAxes, fontsize=9)
    ax.text(0.01, -0.025, f'{len(main_edges):,} internal edges', transform=ax.transAxes, fontsize=9)
    cax = ax.inset_axes([0.55, -0.015, 0.4, 0.029])
    cb = ax.figure.colorbar(sc, cax=cax, orientation='horizontal', ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=8, length=2)
    cb.outline.set_visible(False)
    cb.set_label('Relative exposure strength', fontsize=9, labelpad=2)
    if not hasattr(ax.figure, '_network_colourbar_axes'):
        ax.figure._network_colourbar_axes = []
    ax.figure._network_colourbar_axes.append(cax)

def panel_joint(ax, nodes):
    style(ax, 'Which individuals do the strategies select?', 'B')
    rng = np.random.RandomState(1701)
    x = pd.Categorical(nodes.Age, categories=AGE_VALUES, ordered=True).codes.astype(float)
    x += rng.uniform(-0.26, 0.26, len(nodes))
    groups = [('Neither strategy', NONE), ('Oldest-first only', OLD), ('High-exposure only', HIGH), ('Both strategies', SHARED)]
    for name, color in groups:
        selected = nodes.Recipient_Group == name
        ax.scatter(x[selected], nodes.loc[selected, REL], s=12, color=color, alpha=0.62, linewidths=0, label=f'{name} (n = {selected.sum():,})')
    ax.set_xticks(np.arange(7), AGE_LABELS)
    ax.set(xlabel='Age group (years)', ylabel='Relative exposure strength', xlim=(-0.65, 6.65), ylim=(-0.02, 1.27))
    ax.set_yticks(np.linspace(0, 1, 6))
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 1, 3, 0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper right', ncol=2, frameon=False, fontsize=9, handletextpad=0.2, columnspacing=0.9, markerscale=1.5)
