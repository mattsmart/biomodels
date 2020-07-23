import matplotlib.pyplot as plt
import numpy as np
import os


def image_fancy(image, ax=None, show_labels=False):
    if ax is None:
        plt.figure()
        ax = plt.gca();
    im = ax.imshow(image, interpolation='none', vmin=0, vmax=1, aspect='equal', cmap='gray');

    # Minor ticks
    ax.set_xticks(np.arange(-.5, 28, 1), minor=True);
    ax.set_yticks(np.arange(-.5, 28, 1), minor=True);

    if show_labels:
        # Major ticks
        ax.set_xticks(np.arange(0, 28, 1));
        ax.set_yticks(np.arange(0, 28, 1));
        # Labels for major ticks
        ax.set_xticklabels(np.arange(1, 29, 1));
        ax.set_yticklabels(np.arange(1, 29, 1));
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
    return ax
