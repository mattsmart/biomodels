import matplotlib.pyplot as plt
import numpy as np
import os

from singlecell_simsetup import CELLTYPE_ID, CELLTYPE_LABELS


def plot_proj_timeseries(proj_timeseries_array, num_steps, ensemble, savepath, highlights=None):
    """
    proj_timeseries_array is expected dim p x time
    highlights: either None or a dict of idx: color for certain memory projections to highlight
    """
    assert proj_timeseries_array.shape[0] == len(CELLTYPE_LABELS)
    plt.clf()
    if highlights is None:
        plt.plot(xrange(num_steps), proj_timeseries_array.T, color='blue', linewidth=0.75)
    else:
        plt.plot(xrange(num_steps), proj_timeseries_array.T, color='grey', linewidth=0.55, linestyle='dashed')
        for key in highlights.keys():
            plt.plot(xrange(num_steps), proj_timeseries_array[key,:], color=highlights[key], linewidth=0.75, label=CELLTYPE_LABELS[key])
        plt.legend()
    plt.title('Ensemble mean (n=%d) projection timeseries' % ensemble)
    plt.ylabel('Mean projection onto each memory')
    plt.xlabel('Steps (%d updates, all spins)' % num_steps)
    plt.savefig(savepath)
    return


def plot_basin_occupancy_timeseries(basin_occupancy_timeseries, num_steps, ensemble, threshold, savepath, highlights=None):
    """
    basin_occupancy_timeseries: is expected dim (p + spurious tracked) x time  note spurious tracked default is 'mixed'
    highlights: either None or a dict of idx: color for certain memory projections to highlight
    """
    assert basin_occupancy_timeseries.shape[0] == len(CELLTYPE_LABELS) + 1  # note spurious tracked default is 'mixed'
    plt.clf()
    if highlights is None:
        plt.plot(xrange(num_steps), basin_occupancy_timeseries.T, color='blue', linewidth=0.75)
    else:
        plt.plot(xrange(num_steps), basin_occupancy_timeseries.T, color='grey', linewidth=0.55, linestyle='dashed')
        for key in highlights.keys():
            plt.plot(xrange(num_steps), basin_occupancy_timeseries[key,:], color=highlights[key], linewidth=0.75, label=CELLTYPE_LABELS[key])
        if len(CELLTYPE_LABELS) not in highlights.keys():
            plt.plot(xrange(num_steps), basin_occupancy_timeseries[len(CELLTYPE_LABELS), :], color='orange',
                     linewidth=0.75, label='mixed')
        plt.legend()
    plt.title('Occupancy timeseries (ensemble %d)' % ensemble)
    plt.ylabel('Occupancy in each memory (threshold proj=%.2f)' % threshold)
    plt.xlabel('Steps (%d updates, all spins)' % num_steps)
    plt.savefig(savepath)
    return


def plot_basin_endpoints(endpoint_dict, num_steps, ensemble, savepath, highlights=None):
    """
    endpoint_dict: dict where cell endstates stored via endpoint_dict[idx] = (endpoint_label, projval)
    highlights: either None or a dict of idx: color for certain memory projections to highlight
    """

    # TODO: remove endpoint object maybe and just pass basin occupancy timeseries with a step to this fn?

    # data prep
    occupancies = {}
    for idx in xrange(len(endpoint_dict.keys())):
        endppoint_id, projval = endpoint_dict[idx]
        if endppoint_id in occupancies:
            occupancies[endppoint_id] += 1
        else:
            occupancies[endppoint_id] = 1
    memory_labels = occupancies.keys()
    memory_occupancies = [occupancies[a] for a in memory_labels]
    memory_colors = ['grey' if label not in [CELLTYPE_LABELS[a] for a in highlights.keys()]
                     else highlights[CELLTYPE_ID[label]]
                     for label in memory_labels]
    # plotting
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 12})

    plt.clf()
    fig = plt.figure(1)
    fig.set_size_inches(18.5, 10.5)
    h = plt.bar(xrange(len(memory_labels)), memory_occupancies, color=memory_colors, label=memory_labels)
    plt.subplots_adjust(bottom=0.3)
    xticks_pos = [0.65 * patch.get_width() + patch.get_xy()[0] for patch in h]
    plt.xticks(xticks_pos, memory_labels, ha='right', rotation=45, size=12)
    plt.gca().yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.title('Cell endpoints (%d steps, %d cells)' % (num_steps, ensemble))
    plt.ylabel('Basin occupancy count')
    plt.xlabel('Basin labels')
    fig.savefig(savepath)
    return


def plot_basin_grid(grid_data, ensemble, steps, plotdir, k=1, ax=None, normalize=True, fs=9, relmax=True,
                    rotate_standard=True, extragrid=False):
    """
    plot matrix G_ij of size p x (p + k): grid of data between 0 and 1
    each row represents one of the p encoded basins as an initial condition
    each column represents an endpoint of the simulation starting at a given basin (row)
    G_ij would represent: starting in cell type i, G_ij of the ensemble transitioned to cell type j
    Args:
    - relmax means max of color scale will be data max
    - k represents the number of extra tracked states, by default this is 1 (i.e. mixed state, not in any basin)
    - rotate_standard: determine xlabel orientation
    """
    assert grid_data.shape == (len(CELLTYPE_LABELS), len(CELLTYPE_LABELS) + k)

    assert normalize
    datamax = np.max(grid_data)

    if np.max(grid_data) > 1.0 and normalize:
        grid_data = grid_data / ensemble
        datamax = datamax / ensemble

    if relmax:
        vmax = datamax
    else:
        if normalize:
            vmax = 1.0
        else:
            vmax = ensemble

    if not ax:
        ax = plt.gca()
        plt.gcf().set_size_inches(18.5, 12.5)
    # plot the heatmap
    # note: fix vmax at 0.5 of max works nice
    # note: aspect None, 'auto', scalar, or 'equal'
    imshow_kw = {'cmap': 'YlGnBu', 'aspect': None, 'vmin': 0.0, 'vmax': vmax}  # note: fix at 0.5 of max works nice
    im = ax.imshow(grid_data, **imshow_kw)
    # create colorbar
    cbar_kw = {'aspect': 30, 'pad': 0.02}   # larger aspect, thinner bar
    cbarlabel = 'Basin occupancy fraction'
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=fs+2, labelpad=20)

    # hack title to bottom
    plt.text(0.5, 1.3, 'Basin grid transition data (%d cells per basin, %d steps)' % (ensemble, steps),
             horizontalalignment='center', transform=ax.transAxes, fontsize=fs+4)
    # axis labels
    plt.xlabel('Ensemble fraction after %d steps' % steps, fontsize=fs+2)
    ax.xaxis.set_label_position('top')
    plt.ylabel('Ensemble initial condition (%d cells per basin)' % ensemble, fontsize=fs+2)
    # show all ticks
    ax.set_xticks(np.arange(grid_data.shape[1]))
    ax.set_yticks(np.arange(grid_data.shape[0]))
    # label them with the respective list entries.
    assert k == 1  # TODO col labels as string types + k mixed etc
    ax.set_xticklabels(CELLTYPE_LABELS + ['mixed'], fontsize=fs)
    ax.set_yticklabels(CELLTYPE_LABELS, fontsize=fs)
    # Rotate the tick labels and set their alignment.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    if rotate_standard:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='left')
    else:
        plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
                 rotation_mode="anchor")

    # add gridlines
    ax.set_xticks(np.arange(-.5, grid_data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_data.shape[0], 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)  # grey good to split, white looks nice though

    # hack to add extra gridlines (not clear how to have more than minor and major on one axis)
    if extragrid:
        for xcoord in np.arange(-.5, grid_data.shape[1], 8):
            ax.axvline(x=xcoord, ls='--', color='grey', linewidth=1)
        for ycoord in np.arange(-.5, grid_data.shape[0], 8):
            ax.axhline(y=ycoord, ls='--', color='grey', linewidth=1)

    plt.savefig(plotdir + os.sep + 'plot_basin_grid.pdf', dpi=100, bbox_inches='tight')

    return plt.gca()
