import matplotlib.pyplot as plt
import numpy as np
import os

from class_sweep_cellgraph import SweepCellGraph
from file_io import pickle_load


def visualize_sweep(sweep):

    print("Visualizing data for sweep label:", sweep.sweep_label)
    sweep.printer()
    results = sweep.results_dict

    # A) load classdump
    # B) extract useful info and add to collection
    # Visualize info

    if sweep.k_vary > 2:
        print('sweep.k_vary > 2 not yet supported by visualize_sweep() in load_sweep.py')
        M_of_theta = None
    elif sweep.k_vary == 1:
        param_values = sweep.params_values[0]
        param_name = sweep.params_name[0]
        param_variety = sweep.params_variety[0]

        M_of_theta = np.zeros(sweep.total_runs)
        for run_idx in range(sweep.total_runs):
            run_dir = sweep.sweep_dir + os.sep + '%d' % run_idx
            M_of_theta[run_idx] = results[(run_idx,)]['num_cells']

        plt.plot(param_values, M_of_theta, '--o')
        plt.title(r'$M(\theta)$ for $\theta=$%s' % param_name)
        plt.ylabel('num_cells')
        plt.xlabel('%s' % param_name)
        plt.show()
    else:
        assert sweep.k_vary == 2
        # TODO test this k=2 case
        run_int = 0
        M_of_theta = np.zeros(sweep.sizes)
        for run_id_list in np.ndindex(*sweep.sizes):
            timedir_override = '_'.join([str(i) for i in run_id_list])
            run_dir = sweep.sweep_dir + os.sep + timedir_override
            M_of_theta[run_id_list] = results[run_id_list]['num_cells']
            run_int += 1

        im = plt.imshow(M_of_theta)
        plt.title(r'$M(\{theta})$ for $\{theta}=(%s, %s)$' % (sweep.params_name[0], sweep.params_name[1]))
        plt.ylabel('%s' % sweep.params_name[0])
        plt.xlabel('%s' % sweep.params_name[1])
        plt.colorbar(im)
        plt.show()
    return M_of_theta


if __name__ == '__main__':
    dir_sweep = 'runs' + os.sep + 'sweep_preset_1d_epsilon'
    fpath_pickle = dir_sweep + os.sep + 'sweep.pkl'
    sweep_cellgraph = pickle_load(fpath_pickle)

    visualize_sweep(sweep_cellgraph)
