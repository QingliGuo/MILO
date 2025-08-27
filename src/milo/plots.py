# milo/plots.py

"""Plotting functions for visualizing 83-channel indel mutation profiles."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from . import constants

def plot_id83_profile(
    signature: np.ndarray,
    plot_title: str = "",
    output_path: Path = None
):
    """Generates and saves a plot of an 83-channel indel mutation profile.

    This function creates a bar plot visualizing the counts or proportions
    of mutations across the 83 indel channels, with custom colors and labels
    for different mutation types.

    Args:
        signature (np.ndarray): A numpy array of length 83 containing the
            mutation counts or proportions for each channel.
        plot_title (str, optional): The main title for the plot. Defaults to "".
        output_path (Path, optional): The file path to save the plot. If None,
            the plot is displayed interactively. Defaults to None.
    """
    col_list = []
    color_names1 = ['sandybrown', 'darkorange','yellowgreen','g','peachpuff','coral','orangered',
                    'darkred', 'powderblue', 'skyblue','cornflowerblue','navy']
    for col in color_names1:
        col_list += [col] * 6
    col_list = col_list + ['thistle'] + ['mediumpurple'] * 2 + ['rebeccapurple'] * 3 + ['indigo'] * 5
    col_set = color_names1 + ['thistle','mediumpurple','rebeccapurple','indigo']

    top_label = ['1bp Deletion', '1bp Insertion', '> 1bp Deletion at Repeats \n (Deletion Length)', 
                 '>1bp Insertions at Repeats \n (Insertion Length)', 'Mircohomology \n (Deletion Length)']
    second_top_layer = ['C', 'T', 'C','T','2','3','4','5+','2','3','4','5+','2','3','4','5+']
    second_top_layer_color = ["black"] * 5 + ["white"] * 3 + ["black"] * 3 + ["white", "black"] + ["white"] * 3
    xlabel = ['1','2','3','4','5','6+','1','2','3','4','5','6+','0','1','2','3','4','5+','0','1','2','3','4','5+',
              '1','2','3','4','5','6+','1','2','3','4','5','6+','1','2','3','4','5','6+','1','2','3','4','5','6+',
              '0','1','2','3','4','5+','0','1','2','3','4','5+','0','1','2','3','4','5+','0','1','2','3','4','5+',
              '1','1','2','1','2','3','1','2','3','4','5+']

    sns.set(rc={"figure.figsize":(11, 2.7)})
    sns.set(style="whitegrid", color_codes=True, rc={"grid.linewidth": 0.2, 'grid.color': '.7', 'ytick.major.size': 2,
                                                'axes.edgecolor': '.3', 'axes.linewidth': 1.35,})
    
    fig, ax = plt.subplots()

    ax.bar(range(constants.INDEL_CHANNEL_COUNT), signature, width=0.8, color=col_list)
    ax.set_xticks(range(constants.INDEL_CHANNEL_COUNT))
    ax.set_xticklabels(xlabel, rotation=90, size=7, weight='bold', ha="center", va="center")
    ax.tick_params(axis='y', labelsize=9, pad=1)
    
    if signature.max() > 0:
        ax.set_ylim(0, signature.max() * 1.2)
        ax.annotate(plot_title, (62, signature.max() * 0.92))
        ax.annotate(f'Total = {np.sum(np.abs(signature)):,}', (0, signature.max() * 1.01))
    
    ax.set_ylabel("Counts")
    ax.margins(x=0.007)

    length = [6] * 12 + [1, 2, 3, 5]
    for i, l in enumerate(length):
        left = sum(length[:i]) / 84 + 0.005
        width = l / 84 - 0.001
        
        p_top = plt.Rectangle((left, 1.003), width, 0.14, fill=True, color=col_set[i], transform=ax.transAxes, clip_on=False)
        ax.add_patch(p_top)
        ax.text(left + width/2, 1.07, second_top_layer[i], color=second_top_layer_color[i], weight='bold', ha='center', va='center', transform=ax.transAxes, size=10)

    length2 = [12, 12, 24, 24, 11]
    for i, l in enumerate(length2):
        left = sum(length2[:i]) / 83
        width = l / 83
        p_topmost = plt.Rectangle((left, 1.18), width, 0.1, fill=True, color='w', transform=ax.transAxes, clip_on=False)
        ax.add_patch(p_topmost)
        ax.text(left + width/2, 1.23, top_label[i], ha='center', va='center', transform=ax.transAxes, size=10)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.suptitle(plot_title, y=1.12, fontsize=14)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()

def plot_id83_comparison(
    sig1: np.ndarray,
    sig2: np.ndarray,
    name1: str = "",
    name2: str = "",
    plot_title: str = "",
    output_path: Path = None
):
    """Generates a mirrored bar plot to compare two 83-channel indel profiles.

    This function normalizes two signatures to proportions and plots them
    back-to-back on a vertical axis for direct comparison. It is useful for
    visualizing changes, such as before and after noise correction.

    Args:
        sig1 (np.ndarray): The first signature to plot (e.g., before correction).
        sig2 (np.ndarray): The second signature to plot (e.g., after correction).
        name1 (str, optional): The label for the first signature. Defaults to "".
        name2 (str, optional): The label for the second signature. Defaults to "".
        plot_title (str, optional): The main title for the plot. Defaults to "".
        output_path (Path, optional): The file path to save the plot. If None,
            the plot is displayed interactively. Defaults to None.
    """
    total_sig1 = np.sum(sig1)
    total_sig2 = np.sum(sig2)

    sig1_norm = sig1 / total_sig1 if total_sig1 > 0 else sig1
    sig2_norm = sig2 / total_sig2 if total_sig2 > 0 else sig2
    
    col_list = []
    color_names1 = ['sandybrown', 'darkorange','yellowgreen','g','peachpuff','coral','orangered',
                    'darkred', 'powderblue', 'skyblue','cornflowerblue','navy']
    for col in color_names1:
        col_list += [col] * 6
    col_list = col_list + ['thistle'] + ['mediumpurple'] * 2 + ['rebeccapurple'] * 3 + ['indigo'] * 5
    col_set = color_names1 + ['thistle','mediumpurple','rebeccapurple','indigo']
    
    top_label = ['1bp Deletion', '1bp Insertion', '> 1bp Deletion at Repeats \n (Deletion Length)', 
                 '>1bp Insertions at Repeats \n (Insertion Length)', 'Mircohomology \n (Deletion Length)']
    second_top_layer = ['C', 'T', 'C','T','2','3','4','5+','2','3','4','5+','2','3','4','5+']
    second_top_layer_color = ["black"] * 5 + ["white"] * 3 + ["black"] * 3 + ["white", "black"] + ["white"] * 3
    xlabel = ['1','2','3','4','5','6+','1','2','3','4','5','6+','0','1','2','3','4','5+','0','1','2','3','4','5+',
              '1','2','3','4','5','6+','1','2','3','4','5','6+','1','2','3','4','5','6+','1','2','3','4','5','6+',
              '0','1','2','3','4','5+','0','1','2','3','4','5+','0','1','2','3','4','5+','0','1','2','3','4','5+',
              '1','1','2','1','2','3','1','2','3','4','5+']

    sns.set(rc={"figure.figsize":(11, 4)})
    sns.set(style="whitegrid", color_codes=True, rc={"grid.linewidth": 0.2, 'grid.color': '.7'})
    
    fig, ax = plt.subplots()

    ax.bar(range(constants.INDEL_CHANNEL_COUNT), sig1_norm, width=0.8, color=col_list)
    ax.bar(range(constants.INDEL_CHANNEL_COUNT), -sig2_norm, width=0.8, color=col_list)
    ax.hlines(0, -1, constants.INDEL_CHANNEL_COUNT, linestyle='dashed', alpha=0.5)
    
    ax.set_xticks(range(constants.INDEL_CHANNEL_COUNT))
    ax.set_xticklabels(xlabel, rotation=90, size=7, weight='bold', ha="center", va="center")
    ax.tick_params(axis='y', labelsize=9, pad=1)

    max_val = max(np.max(sig1_norm), np.max(sig2_norm)) if total_sig1 > 0 or total_sig2 > 0 else 0
    if max_val > 0:
        ax.set_ylim(-max_val * 1.3, max_val * 1.3)
        ax.annotate(name1, (52, max_val * 0.5), fontsize=12)
        ax.annotate(name2, (52, -max_val * 0.8), fontsize=12)
        ax.annotate(f'Total = {total_sig1:,}', (0.5, max_val * 1.01), fontsize=9)
        ax.annotate(f'Total = {total_sig2:,}', (0.5, -max_val * 1.22), fontsize=9)
    
    ax.set_ylabel("Proportion", size=13)
    ax.margins(x=0.007)
    
    length = [6] * 12 + [1, 2, 3, 5]
    for i, l in enumerate(length):
        left = sum(length[:i]) / 84 + 0.005
        width = l / 84 - 0.001
        p_top = plt.Rectangle((left, 1.02), width, 0.08, fill=True, color=col_set[i], transform=ax.transAxes, clip_on=False)
        ax.add_patch(p_top)
        ax.text(left + width/2, 1.06, second_top_layer[i], color=second_top_layer_color[i], weight='bold', ha='center', va='center', transform=ax.transAxes, size=10)

    length2 = [12, 12, 24, 24, 11]
    for i, l in enumerate(length2):
        left = sum(length2[:i]) / 83
        width = l / 83
        p_topmost = plt.Rectangle((left, 1.12), width, 0.08, fill=True, color='w', transform=ax.transAxes, clip_on=False)
        ax.add_patch(p_topmost)
        ax.text(left + width/2, 1.16, top_label[i], ha='center', va='center', transform=ax.transAxes, size=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.suptitle(plot_title, y=1.0, fontsize=14)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()
