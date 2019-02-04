import matplotlib.pyplot as plt
import numpy as np
import os

from settings import FOLDER_OUTPUT


def plot_matrix(arr, method='U', title_mod='', show=False, rotate_standard=True, fs=10, ax=None, cmap_int=11,
                xlabels=None, ylabels=None, nolabels=False, plotdir=FOLDER_OUTPUT):
    assert len(arr.shape) == 2
    if ax is None:
        f = plt.figure(figsize=(18.5, 12.5))
        ax = plt.gca()
    # plot the heatmap
    """
    choices: 'YlGnBu', 'bone_r', 'BuPu', 'PuBuGn', 'Greens', 'Spectral', 'Spectral_r', 'cubehelix_r', 'magma_r'
    note: aspect None, 'auto', scalar, or 'equal'
    """
    if cmap_int is not None:
        cmap = plt.get_cmap('YlGnBu', cmap_int)
    else:
        cmap = plt.get_cmap('YlGnBu')
    imshow_kw = {'cmap': cmap, 'aspect': None, 'vmin': np.min(arr), 'vmax': np.max(arr)}
    im = ax.imshow(arr, **imshow_kw)
    # title
    plt.title('Matrix plot - %s %s' % (method, title_mod), y=1.08)
    # plt.text(0.5, 1.3, 'Matrix plot - %s %s' % (method, title_mod), horizontalalignment='center',
    #         transform=ax.transAxes, fontsize=fs+4)
    # create colorbar
    cbar_kw = {'aspect': 30, 'pad': 0.02}   # larger aspect, thinner bar
    cbarlabel = 'Matrix elements'
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=fs+2, labelpad=20)
    # axis labels
    ax.xaxis.set_label_position('top')
    # show all ticks
    if not nolabels:
        ax.set_xticks(np.arange(arr.shape[1]))
        ax.set_yticks(np.arange(arr.shape[0]))
    # label them with the respective list entries.
    if xlabels is not None:
        ax.set_xticklabels(xlabels, fontsize=fs)
    if ylabels is not None:
        ax.set_yticklabels(ylabels, fontsize=fs)
    # rotate the tick labels and set their alignment
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    if rotate_standard:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='left')
    else:
        plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
                 rotation_mode="anchor")
    # add gridlines
    ax.set_xticks(np.arange(-.5, arr.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, arr.shape[0], 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)  # grey good to split, white looks nice though
    # save and show
    plt.savefig(plotdir + os.sep + 'matrix_%s_%s.png' % (method, title_mod), bbox_inches='tight')
    if show:
        plt.show()
    plt.close('all')
    return


if __name__ == '__main__':
    arr = -3+ np.random.rand(100, 100)
    plot_matrix(arr, show=True, fs=6, nolabels=True)
