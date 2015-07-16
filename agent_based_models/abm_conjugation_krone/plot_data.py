import matplotlib.pyplot as plt

# note:
# dict keys are iters, time, E, R, D, T, N

def data_plotter(lattice_dict, datafile_dir, plot_dir):

    total_cells = lattice_dict['E'][0] + lattice_dict['R'][0] + lattice_dict['D'][0] + lattice_dict['T'][0]  # total bacteria
    n = int(total_cells**0.5)

    plt.figure(1)

    plt.plot(lattice_dict['time'], lattice_dict['E'], label='Empty lattice points')
    plt.plot(lattice_dict['time'], lattice_dict['R'], label='Receivers')
    plt.plot(lattice_dict['time'], lattice_dict['D'], label='Donors')
    plt.plot(lattice_dict['time'], lattice_dict['T'], label='Transconjugants')
    plt.plot(lattice_dict['time'], lattice_dict['N'], label='Nutrients')

    ax = plt.gca()
    ax.set_title('Cell and Nutrient Populations over time (n = %d)' % n)
    ax.set_ylabel('Number of cells')
    ax.set_xlabel('Time (h)')

    plt.legend()

    f = plt.gcf()
    f.set_size_inches(20.0, 8.0)  # alternative: 20.0, 8.0
    f.tight_layout()
    plt.savefig(plot_dir + 'population_vs_time' + '.png')
    plt.clf()
    
    return

