import matplotlib.pyplot as plt


def plasmid_stats(lattice, dict_counts):

    keys = ['R', 'D', 'T']
    plasmid_counts_by_type = {key: [0] * dict_counts[key] for key in keys}
    cell_type_iterator = {key: 0 for key in keys}

    # get distribution
    n = len(lattice)
    for i in xrange(n):
        for j in xrange(n):
            cell = lattice[i][j]
            if cell.label != '_':
                idx = cell_type_iterator[cell.label]
                plasmid_counts_by_type[cell.label][idx] = cell.plasmid_amensal
                cell_type_iterator[cell.label] += 1

    return plasmid_counts_by_type


def plasmid_plotter(plasmid_counts, plot_path):
    if len(plasmid_counts) > 0:
        total_cells = len(plasmid_counts)
        f = plt.figure()
        plt.hist(plasmid_counts)
        ax = plt.gca()
        ax.set_title('Plasmid Count Histogram (cells = %d)' % total_cells)
        ax.set_ylabel('Number of cells')
        ax.set_xlabel('Plasmid Count')
        f.set_size_inches(20.0, 8.0)  # alternative: 20.0, 8.0
        f.tight_layout()
        plt.savefig(plot_path)
        plt.clf()
    return


def plasmid_plotter_wrapper(lattice, dict_counts, time, plot_dir):
    plasmid_counts_by_type = plasmid_stats(lattice, dict_counts)
    plasmid_plotter(plasmid_counts_by_type['R'], plot_dir + 'R_plasmid_histogram_at_time_%f' % time + '.png')
    plasmid_plotter(plasmid_counts_by_type['D'], plot_dir + 'D_plasmid_histogram_at_time_%f' % time + '.png')
    plasmid_plotter(plasmid_counts_by_type['T'], plot_dir + 'T_plasmid_histogram_at_time_%f' % time + '.png')
    return
