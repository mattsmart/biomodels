import matplotlib.pyplot as plt
import os


def data_plotter(lattice_dict, datafile_dir, plot_dir):

    # total spaces on grid implies grid size
    total_cells = lattice_dict['E'][0] + lattice_dict['D_a'][0] + lattice_dict['D_b'][0] + lattice_dict['B'][0]
    n = int(total_cells**0.5)

    plt.figure(1)

    plt.plot(lattice_dict['time'], lattice_dict['E'], label='Empty lattice points')
    plt.plot(lattice_dict['time'], lattice_dict['D_a'], label='Donors (Type A)')
    plt.plot(lattice_dict['time'], lattice_dict['D_b'], label='Donors (Type B)')
    plt.plot(lattice_dict['time'], lattice_dict['B'], label='Debris')

    ax = plt.gca()
    ax.set_title('Cell Populations over time (n = %d)' % n)
    ax.set_ylabel('Number of cells')
    ax.set_xlabel('Time (h)')

    plt.legend()

    f = plt.gcf()
    f.set_size_inches(20.0, 8.0)  # alternative: 20.0, 8.0
    f.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'population_vs_time.png'))
    plt.clf()
    
    return
