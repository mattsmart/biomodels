import matplotlib.pyplot as plt
import numpy as np
import os

from class_sweep_cellgraph import SweepCellGraph
from file_io import pickle_load
from utils_networkx import check_tree_isomorphism, draw_from_adjacency


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

        # Part 1) plot num of cells as function of varying parameter
        M_of_theta = np.zeros(sweep.total_runs)
        for run_idx in range(sweep.total_runs):
            run_dir = sweep.sweep_dir + os.sep + '%d' % run_idx
            M_of_theta[run_idx] = results[(run_idx,)]['num_cells']

        plt.plot(param_values, M_of_theta, '--o')
        plt.title(r'$M(\theta)$ for $\theta=$%s' % param_name)
        plt.ylabel('num_cells')
        plt.xlabel('%s' % param_name)
        fpath = sweep.sweep_dir + os.sep + 'num_cells_1d_vary_%s.pdf' % (param_name)
        plt.savefig(fpath)
        plt.show()

        # Part 2) plot adjacency matrix variety TODO - on top of the 1D, M(theta) plot ?
        A_uniques = {}
        for run_idx in range(sweep.total_runs):
            A_run = results[(run_idx,)]['adjacency']
            num_cells = results[(run_idx,)]['num_cells']
            # Case A - have we seen a graph of this size yet? if no it's unique
            if num_cells not in A_uniques.keys():
                A_id = 'M%s_v0' % num_cells
                A_uniques[num_cells] = {A_id: {'adjacency': A_run, 'run': run_idx}}
            else:
                # Need now to compare the adjacency matrix to those observed before, to see if it is unique
                # - necessary condition for uniqueness is unique Degree matrix, D -- if D is unique, then A is unique
                is_iso = False
                for k, v in A_uniques[num_cells].items():
                    is_iso, iso_swaps = check_tree_isomorphism(A_run, v['adjacency'])
                    if is_iso:
                        break
                if not is_iso:
                    nunique = len(A_uniques[num_cells].keys())
                    A_id = 'M%s_v%d' % (num_cells, nunique)
                    A_uniques[num_cells][A_id] = {'adjacency': A_run, 'run': run_idx}

        keys_sorted = sorted(A_uniques.keys())
        for k in keys_sorted:
            q = len(A_uniques[k].keys())
            print("Num cells: %d observed %d unique adjacencies" % (k, q))
            if q > 1:
                print("\t", A_uniques[k].keys())
            for unique_graph_id in A_uniques[k].keys():
                A_run = A_uniques[k][unique_graph_id]['adjacency']
                run_idx = A_uniques[k][unique_graph_id]['run']
                fpath = sweep.sweep_dir + os.sep + 'run_%d_id_%s.jpg' % (run_idx, unique_graph_id)
                degree = np.diag(np.sum(A_run, axis=1))
                draw_from_adjacency(A_run, node_color=np.diag(degree), labels=None, cmap='Pastel1',
                                    title=unique_graph_id, spring=False, seed=None, fpath=fpath)
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
