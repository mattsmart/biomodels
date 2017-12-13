import matplotlib.pyplot as plt
import os
from math import pi

from singlecell_simsetup import CELLTYPE_LABELS


def plot_as_radar(projection_vec, memory_labels=CELLTYPE_LABELS):
    """
    # radar plots not built-in to matplotlib
    # reference code uses pandas: https://python-graph-gallery.com/390-basic-radar-chart/
    """

    # number of variable
    p = len(memory_labels)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(p) * 2 * pi for n in range(p)]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    # Draw one ax per variable + add labels labels yet
    plt.xticks(angles, memory_labels, color='grey', size=6)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], ["-1.0", "-0.5", "0.0", "0.5", "1.0"], color="grey", size=7)
    plt.ylim(-1, 1)

    # Plot data
    ax.plot(angles, projection_vec, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, projection_vec, 'b', alpha=0.1)
    return fig, ax


def save_manual(fig, dir, fname, close=True):
    filepath = dir + os.sep + fname + ".png"
    fig.savefig(filepath, dpi=100)
    if close:
        plt.close()
    return
