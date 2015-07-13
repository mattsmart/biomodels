import matplotlib.pyplot as plt

# note:
# dict keys are iters, time, E, R, D

def data_plotter(lattice_dict, datafile_dir, plot_dir):

    N = lattice_dict['E'][0] + lattice_dict['R'][0] + lattice_dict['D'][0]
    n = int(N**0.5)

    plt.figure(1)

    plt.plot(lattice_dict['time'], lattice_dict['E'], label='Empty lattice points')
    plt.plot(lattice_dict['time'], lattice_dict['R'], label='Receivers')
    plt.plot(lattice_dict['time'], lattice_dict['D'], label='Donors')

    ax = plt.gca()
    ax.set_title('Cell Populations over time (n = %d)' % n)
    ax.set_ylabel('Number of cells')
    ax.set_xlabel('Time (h)')

    plt.legend()

    f = plt.gcf()
    f.set_size_inches(10.0, 4.0)  # alternative: 20.0, 8.0
    f.tight_layout()
    plt.savefig(plot_dir + 'population_vs_time' + '.jpg')
    plt.clf()
    
    return
