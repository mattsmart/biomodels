import singlecell.init_multiprocessing  # BEFORE numpy

import numpy as np
import os
import matplotlib.pyplot as plt

from twocell_simulate import twocell_sim_fast, twocell_sim_as_onelargemodel
from multicell.multicell_class import SpatialCell
from singlecell.singlecell_constants import MEMS_UNFOLD, BETA, RUNS_FOLDER
from singlecell.singlecell_simsetup import singlecell_simsetup


def twocell_ensemble_stats(simsetup, steps, beta, gamma, ens=10, monolothic_flag=False):
    overlap_data = np.zeros((ens, 2 * simsetup['P']))
    assert simsetup['P'] == 1

    XI_scaled = simsetup['XI'] / simsetup['N']

    def random_twocell_lattice():
        cell_a_init = np.array([2*int(np.random.rand() < .5) - 1 for _ in xrange(simsetup['N'])]).T
        cell_b_init = np.array([2*int(np.random.rand() < .5) - 1 for _ in xrange(simsetup['N'])]).T
        lattice = [[SpatialCell(cell_a_init, 'Cell A', [0, 0], simsetup),
                    SpatialCell(cell_b_init, 'Cell B', [0, 1],
                                simsetup)]]  # list of list to conform to multicell slattice funtions
        return lattice

    for traj in xrange(ens):
        if traj % 100 == 0:
            print "Running traj", traj, "..."
        lattice = random_twocell_lattice()

        # TODO replace with twocell_sim_as_onelargemodel (i.e. one big ising model)
        if monolothic_flag:
            lattice = twocell_sim_as_onelargemodel(lattice, simsetup, steps, beta=beta, gamma=gamma)
        else:
            lattice = twocell_sim_fast(lattice, simsetup, steps, beta=beta, exostring='no_exo_field',
                                       gamma=gamma, app_field=None, app_field_strength=0.0)
        cell_A_endstate = lattice[0][0].get_state_array()[:,-1]
        cell_B_endstate = lattice[0][1].get_state_array()[:,-1]
        cell_A_overlaps = np.dot(XI_scaled.T, cell_A_endstate)
        cell_B_overlaps = np.dot(XI_scaled.T, cell_B_endstate)
        overlap_data[traj, 0:simsetup['P']] = cell_A_overlaps
        overlap_data[traj, simsetup['P']:] = cell_B_overlaps

    if simsetup['P'] == 1:
        plt.figure()
        plt.scatter(overlap_data[:,0], overlap_data[:,1], alpha=0.2)
        fname = "overlaps_ens%d_beta%.2f_gamma%.2f_mono%d.png" % (ens, beta, gamma, monolothic_flag)
        plt.title(fname)
        plt.xlabel(r"$m_A$")
        plt.ylabel(r"$m_B$")
        plt.savefig(RUNS_FOLDER + os.sep + "twocell_analysis" + os.sep + fname)

        """
        import seaborn as sns; sns.set()
        import pandas as pd
        df_overlap_data = pd.DataFrame({r"$m_A$":overlap_data[:,0], r"$m_B$":overlap_data[:,1]})
        
        cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        #ax = sns.scatterplot(x=r"$m_A$", y=r"$m_B$", palette=cmap,
        #                     sizes=(20, 200), hue_norm=(0, 7), legend="full", data=df_overlap_data)
        ax = sns.kdeplot(overlap_data[:,0], overlap_data[:,1], shade=True, palette=cmap)
        plt.show()
        """

    else:
        # TODO do some dim reduction
        assert 1==2
    return overlap_data


if __name__ == '__main__':
    random_mem = False
    random_W = False
    simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, npzpath=MEMS_UNFOLD,
                                   curated=True)
    print 'note: N =', simsetup['N']

    ensemble = 500
    steps = 20
    beta = 2.0  # 2.0
    #gamma = 1.0
    for gamma in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10000.0]:
        twocell_ensemble_stats(simsetup, steps, beta, gamma, ens=ensemble, monolothic_flag=False)
        twocell_ensemble_stats(simsetup, steps, beta, gamma, ens=ensemble, monolothic_flag=True)
