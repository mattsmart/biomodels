import matplotlib.pyplot as plt

# note dict keys are: iters, time, E, R, D

def data_plotter(grid_dict, datafile_dir, plot_dir):

    N = grid_dict['E'][0] + grid_dict['R'][0] + grid_dict['D'][0]
    n = int(N**0.5)

    plt.figure(1)

    plt.plot(grid_dict['time'],grid_dict['E'],label='Empty grid cells')
    plt.plot(grid_dict['time'],grid_dict['R'],label='Receivers')
    plt.plot(grid_dict['time'],grid_dict['D'],label='Donors')

    ax = plt.gca()
    ax.set_title('Cell Populations over time (n = %d)' % n)
    ax.set_ylabel('Number of cells')
    ax.set_xlabel('Time (h)')

    plt.legend()

    f = plt.gcf()
    f.set_size_inches(10.0,4.0)     #20.0, 8.0
    f.tight_layout()
    plt.savefig(plot_dir + 'population_vs_time' + '.jpg')
    plt.clf()
    
    return

