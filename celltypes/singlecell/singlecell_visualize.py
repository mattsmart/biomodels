import matplotlib.pyplot as plt
import numpy as np
import os
from math import pi

from singlecell_simsetup import CELLTYPE_LABELS


def plot_as_bar(projection_vec, memory_labels=CELLTYPE_LABELS):
    fig = plt.figure(1)
    fig.set_size_inches(18.5, 10.5)
    h = plt.bar(xrange(len(memory_labels)), projection_vec, label=memory_labels)
    plt.subplots_adjust(bottom=0.3)
    xticks_pos = [0.65 * patch.get_width() + patch.get_xy()[0] for patch in h]
    plt.xticks(xticks_pos, memory_labels, ha='right', rotation=45, size=7)
    return fig, plt.gca()


def plot_as_radar(projection_vec, memory_labels=CELLTYPE_LABELS):
    """
    # radar plots not built-in to matplotlib
    # reference code uses pandas: https://python-graph-gallery.com/390-basic-radar-chart/
    """

    # number of variable
    p = len(memory_labels)

    # Angle of each axis in the plot
    angles = [n / float(p) * 2 * pi for n in range(p)]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    # Draw one ax per variable + add labels
    plt.xticks(angles, memory_labels, color='grey', size=12)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], ["-1.0", "-0.5", "0.0", "0.5", "1.0"], color="grey", size=12)
    plt.ylim(-1, 1)

    # Plot data
    ax.plot(angles, projection_vec, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, projection_vec, 'b', alpha=0.1)

    # Rotate the type labels
    plt.gcf().canvas.draw()
    angles = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()) + 1)
    angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
    angles = np.rad2deg(angles)
    labels=[]
    for label, angle in zip(ax.get_xticklabels(), angles):
        x, y = label.get_position()
        lab = ax.text(x, y - 0.05, label.get_text(), transform=label.get_transform(),
                      ha=label.get_ha(), va=label.get_va(), size=11.5)
        lab.set_rotation(angle)
        labels.append(lab)
    ax.set_xticklabels([])

    return fig, ax


def save_manual(fig, dir, fname, close=True):
    filepath = dir + os.sep + fname + ".png"
    fig.savefig(filepath, dpi=100)
    if close:
        plt.close()
    return
