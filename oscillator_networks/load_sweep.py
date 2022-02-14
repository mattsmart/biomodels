import matplotlib.pyplot as plt
import numpy as np
import os

from class_sweep_cellgraph import SweepCellGraph
from utils_io import pickle_load
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

        # Part 1) get num of cells as function of varying parameter
        M_of_theta = np.zeros(sweep.total_runs)
        for run_idx in range(sweep.total_runs):
            run_dir = sweep.sweep_dir + os.sep + '%d' % run_idx
            M_of_theta[run_idx] = results[(run_idx,)]['num_cells']

        # Part 2) plot adjacency matrix variety
        A_uniques = {}
        iso_labels_to_run_idx = {}
        """
        Structure of A_uniques: dict
            key: num_cells
            value: a dict of
                key: A_id, e.g. "7_v0" means 7 cell graph. variant 0 (enumerating observed non-isomorphic 7 cell graphs)
                value: {'adjacency': np array,
                        'runs': list of run indices where this variant was observed,
                        'iso': integer defining the variant id}
        Structure of iso_labels_to_run_idx: dict
            maps variant int id to list of run indices; used mainly for annotating plots
            int -> list of run indices 
        """
        for run_idx in range(sweep.total_runs):
            A_run = results[(run_idx,)]['adjacency']
            num_cells = results[(run_idx,)]['num_cells']
            # Case A - have we seen a graph of this size yet? if no it's unique
            if num_cells not in A_uniques.keys():
                A_id = 'M%s_v0' % num_cells
                A_uniques[num_cells] = {A_id: {'adjacency': A_run, 'runs': [run_idx], 'iso': 0}}
                iso_labels_to_run_idx[0] = [run_idx]
            else:
                # Need now to compare the adjacency matrix to those observed before, to see if it is unique
                # - necessary condition for uniqueness is unique Degree matrix, D -- if D is unique, then A is unique
                is_iso = False
                for k, v in A_uniques[num_cells].items():
                    is_iso, iso_swaps = check_tree_isomorphism(A_run, v['adjacency'])
                    if is_iso:
                        A_uniques[num_cells][k]['runs'] += [run_idx]
                        iso_labels_to_run_idx[v['iso']] += [run_idx]
                        break
                if not is_iso:
                    nunique = len(A_uniques[num_cells].keys())
                    A_id = 'M%s_v%d' % (num_cells, nunique)
                    A_uniques[num_cells][A_id] = {'adjacency': A_run, 'runs': [run_idx], 'iso': nunique}
                    if nunique in iso_labels_to_run_idx.keys():
                        iso_labels_to_run_idx[nunique] += [run_idx]
                    else:
                        iso_labels_to_run_idx[nunique] = [run_idx]

        keys_sorted = sorted(A_uniques.keys())
        print('Plotting unique graphs: nodes are colored by degree')
        for k in keys_sorted:
            q = len(A_uniques[k].keys())
            print("Num cells: %d observed %d unique adjacencies" % (k, q))
            if q > 1:
                print("\t", A_uniques[k].keys())
            for unique_graph_id in A_uniques[k].keys():
                A_run = A_uniques[k][unique_graph_id]['adjacency']
                run_idx = A_uniques[k][unique_graph_id]['runs'][0]
                fpath = sweep.sweep_dir + os.sep + 'run_%d_id_%s.jpg' % (run_idx, unique_graph_id)
                degree = np.diag(np.sum(A_run, axis=1))
                draw_from_adjacency(A_run, node_color=np.diag(degree), labels=None, cmap='Pastel1',
                                    title=unique_graph_id, spring=False, seed=None, fpath=fpath)
        """
        # Alternate block to simply plot adjacency from all runs, ignoring uniqueness 
        for run_idx in range(sweep.total_runs):
            A_run = results[(run_idx,)]['adjacency']
            fpath = sweep.sweep_dir + os.sep + 'run_%d.jpg' % (run_idx)
            title = r'$\epsilon = %.3f$ (%d cells)' % (param_values[run_idx], A_run.shape[0])
            degree = np.diag(np.sum(A_run, axis=1))
            draw_from_adjacency(A_run, node_color=np.diag(degree), labels=None, cmap='Pastel1',
                                title=title, spring=False, seed=None, fpath=fpath, figsize=(8,8))
        """
        print("iso_labels_to_run_idx:")
        print(iso_labels_to_run_idx)
        # Create fancier M(theta) plot using unique graphs as diff colors
        kwargs = dict(
            markersize=6,
            markeredgecolor='k',
            markeredgewidth=0.4,
        )
        plt.figure(figsize=(4, 4), dpi=600)  # (4,4) used for 100 diffusion 1D
        plt.plot(param_values, M_of_theta, '--o', **kwargs)
        # plot separate points on top for the special runs with graph isomorphisms
        for iso in sorted(iso_labels_to_run_idx.keys()):
            if iso > 0:
                specific_runs = iso_labels_to_run_idx[iso]
                plt.plot(param_values[specific_runs], M_of_theta[specific_runs], 'o', **kwargs)
                plt.plot()
        plt.title(r'$M(\theta)$ for $\theta=$%s' % param_name)
        plt.ylabel('num_cells')
        plt.xlabel('%s' % param_name)
        fpath = sweep.sweep_dir + os.sep + 'num_cells_1d_vary_%s.pdf' % (param_name)
        plt.savefig(fpath)
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
    #sweep_dir = 'sweeps' + os.sep + 'sweep_preset_1d_epsilon_ndiv_bam'
    sweep_dir = 'runs' + os.sep + 'sweep_preset_1d_epsilon'
    #sweep_dir = 'runs' + os.sep + 'sweep_preset_2d_diffusion_ndiv_bam_test'
    fpath_pickle = sweep_dir + os.sep + 'sweep.pkl'
    sweep_cellgraph = pickle_load(fpath_pickle)

    # hotswap directory attribute in case sweep folder was archived
    if sweep_dir != sweep_cellgraph.sweep_dir:
        sweep_cellgraph.sweep_dir = sweep_dir

    visualize_sweep(sweep_cellgraph)
