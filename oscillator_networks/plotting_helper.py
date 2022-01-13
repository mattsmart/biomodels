import matplotlib.pyplot as plt
from os import sep


# PLOTTING CONSTANTS
PLT_SAVE = 'default'
FLAG_SAVE = True
BBOX_INCHES = None
FLAG_SHOW = True
FLAG_TABLE = True
LOC_TABLE = 'center right'
BBOX_TABLE = None


def plot_options_build(alloff=False, **plot_options):
    main_keys = ['flag_show', 'flag_save', 'flag_table']
    plot_options = {'flag_table': plot_options.get('flag_table', FLAG_TABLE),
                    'loc_table': plot_options.get('loc_table', LOC_TABLE),
                    'bbox_table': plot_options.get('bbox_table', BBOX_TABLE),
                    'plt_save': plot_options.get('plt_save', PLT_SAVE),
                    'flag_save': plot_options.get('flag_save', FLAG_SAVE),
                    'bbox_inches': plot_options.get('bbox_inches', BBOX_INCHES),
                    'flag_show': plot_options.get('flag_show', FLAG_SHOW)}
    if alloff:
        plot_options = {k: False for k in main_keys}
    return plot_options


def plot_handler(fig, ax, params, plot_options=None):
    if plot_options is None:
        plot_options = plot_options_build()

    if plot_options.get('flag_table', FLAG_TABLE):
        loc = plot_options.get('loc_table', LOC_TABLE)
        bbox = plot_options.get('bbox_table', BBOX_TABLE)
        plot_table_params(ax, params, loc=loc, bbox=bbox)

    if plot_options.get('flag_save', FLAG_SAVE):
        savename = OUTPUT_DIR + sep + plot_options.get('plt_save') + '.pdf'
        bbox_inches = plot_options.get('bbox_inches', BBOX_INCHES)
        fig.savefig(savename, bbox_inches=bbox_inches)

    if plot_options.get('flag_show', FLAG_SHOW):
        plt.show()
    return fig, ax


def plot_table_params(ax, params, loc=LOC_TABLE, bbox=None):
    """
    params is Params object
    loc options 'center right', 'best'
    bbox is x0, y0, height, width e.g. (1.1, 0.2, 0.1, 0.75)
    """
    # create table of params
    row_labels = ['system', 'feedback']
    row_labels += [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params.system], [params.feedback]]
    table_vals += [[val] for val in params.params_list]  # note weird format
    # plot table
    param_table = ax.table(cellText=table_vals,
                           colWidths=[0.1]*3,
                           rowLabels=row_labels,
                           loc=loc, bbox=bbox)
    #ax.text(12, 3.4, 'Params', size=8)
    return ax
