import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
import joblib
import pandas as pd
import umap
import time

import plotly
import plotly.express as px

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

from utils.file_io import RUNS_FOLDER

"""
# path hack for relative import in jupyter notebook
# LIBRARY GLOBAL MODS
CELLTYPES = os.path.dirname(os.path.abspath(''))
sys.path.append(CELLTYPES)"""

"""
This is .py form of the original .ipynb for exploring UMAP of the multicell dataset 
Main data structure: dict of dicts (called data_subdicts)
Structure is
    datasets[idx]['data'] = X
    datasets[idx]['index'] = list(range(num_runs))
    datasets[idx]['energies'] = X_energies
    datasets[idx]['num_runs'] = num_runs
    datasets[idx]['total_spins'] = total_spins
    datasets[idx]['multicell_template'] = multicell_template
    datasets[idx]['reducer'] = umap.UMAP(**umap_kwargs)
    datasets[idx]['reducer'].fit(X)
    datasets[idx]['embedding'] = datasets[idx]['reducer'].transform(X)
    
Here, each data subdict is pickled as a data_subdict pickle object
Regular location: 
    multicell_manyruns / gamma20.00e_10k / dimreduce / [files]
    files include dimreduce.pkl 
"""


# these set the defaults for modifications introduced inmain
UMAP_SEED = 100
# see defaults: https://umap-learn.readthedocs.io/en/latest/api.html
UMAP_KWARGS = {
    'random_state': UMAP_SEED,
    'n_components': 3,
    'metric': 'euclidean',
    'init': 'spectral',
    'min_dist': 0.1,
    'spread': 1.0,
}


def generate_control_data(total_spins, num_runs):
    X_01 = np.random.randint(2, size=(num_runs, total_spins))
    X = X_01 * 2 - 1
    return X


def make_dimreduce_object(data_subdict, umap_kwargs=UMAP_KWARGS):
    manyruns_path = data_subdict['path']
    fpath_state = manyruns_path + os.sep + 'aggregate' + os.sep + 'X_aggregate.npz'
    fpath_energy = manyruns_path + os.sep + 'aggregate' + os.sep + 'X_energy.npz'
    fpath_pickle = manyruns_path + os.sep + 'multicell_template.pkl'
    print(fpath_state)
    X = np.load(fpath_state)['arr_0'].T  # umap wants transpose
    X_energies = np.load(fpath_energy)['arr_0'].T  # umap wants transpose (?)
    with open(fpath_pickle, 'rb') as pickle_file:
        multicell_template = pickle.load(pickle_file)  # unpickling multicell object

    # store data and metadata in datasets object
    num_runs, total_spins = X.shape
    print(X.shape)
    datasets[idx]['data'] = X
    datasets[idx]['index'] = list(range(num_runs))
    datasets[idx]['energies'] = X_energies
    datasets[idx]['num_runs'] = num_runs
    datasets[idx]['total_spins'] = total_spins
    datasets[idx]['multicell_template'] = multicell_template  # not needed? stored already

    # perform dimension reduction
    t1 = time.time()
    datasets[idx]['reducer'] = umap.UMAP(**umap_kwargs)
    datasets[idx]['reducer'].fit(X)
    datasets[idx]['embedding'] = datasets[idx]['reducer'].transform(X)
    print('Time to fit: %.2f sec' % (time.time() - t1))

    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert (np.all(datasets[idx]['embedding'] == datasets[idx]['reducer'].embedding_))
    print('embedding.shape:', datasets[idx]['embedding'].shape)

    return data_subdict


def save_dimreduce_object(data_subdict, savepath, flag_joblib=True, compress=3):
    from pathlib import Path
    parent = Path(savepath).parent
    if not os.path.exists(parent):
        os.makedirs(parent)
    if flag_joblib:
        assert savepath[-2:] == '.z'
        with open(savepath, 'wb') as fp:
            joblib.dump(data_subdict, fp, compress=compress)
    else:
        assert savepath[-4:] == '.pkl'
        with open(savepath, 'wb') as fp:
            pickle.dump(data_subdict, fp)
    return


def plot_umap_of_data_nonBokeh(data_subdict):
    num_runs = data_subdict['num_runs']
    label = data_subdict['label']
    embedding = data_subdict['embedding']
    c = data_subdict['energies'][:, 0]  # range(num_runs)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=c, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.colorbar()
    plt.title('UMAP projection of the %s dataset' % label, fontsize=24)
    return


def umap_plotly_express(data_subdict):
    """
    Supports 2D and 3D embeddings
    """

    num_runs = data_subdict['num_runs']
    label = data_subdict['label']
    embedding = data_subdict['embedding']
    reducer = data_subdict['reducer']
    dirpath = data_subdict['path'] + os.sep + 'dimreduce'
    c = data_subdict['energies'][:, 0]  # range(num_runs)

    umap_dim = embedding.shape[1]
    assert umap_dim in [2, 3]

    if umap_dim == 2:
        df = pd.DataFrame({'index': range(num_runs),
                           'energy': c,
                           'x': embedding[:, 0],
                           'y': embedding[:, 1]})

        fig = px.scatter(df, x='x', y='y',
                         color='energy',
                         title='UMAP of %s dataset' % label)

    else:
        df = pd.DataFrame({'index': range(num_runs),
                           'energy': c,
                           'x': embedding[:, 0],
                           'y': embedding[:, 1],
                           'z': embedding[:, 2]})

        fig = px.scatter_3d(df, x='x', y='y', z='z',
                            color='energy',
                            title='UMAP of %s dataset' % label)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    fig.write_html(dirpath + os.sep + "umap_plotly_%s.html" % label)
    fig.show()
    return


if __name__ == '__main__':
    build_dimreduce_dicts = True
    add_control_data = True
    vis_all = True

    # Step 0) which 'manyruns 'dirs to work with
    # gamma_vals = ['0e','0.05e', '0.1e', '0.2e', '1e', '2e','20e']
    #manyruns_dirnames = ['gamma0.00_10k', 'gamma0.05_10k']
    manyruns_dirnames = ['gamma0.00_10k', 'gamma0.05_10k', 'gamma0.10_10k', 'gamma0.20_10k',
                         'gamma1.00_10k', 'gamma2.00_10k', 'gamma20.00_10k']

    manyruns_paths = [RUNS_FOLDER + os.sep + 'multicell_manyruns' + os.sep + dirname
                      for dirname in manyruns_dirnames]

    # Step 1) umap kwargs
    umap_kwargs = UMAP_KWARGS.copy()
    umap_kwargs['n_components'] = 3  # TODO don't need to spec 'live', can embed later?

    # Step 2) make/load data
    datasets = {i: {'label': manyruns_dirnames[i],
                    'path': manyruns_paths[i]}
                for i in range(len(manyruns_dirnames))}

    for idx in range(len(manyruns_dirnames)):
        print('Dim. reduction on manyruns: %s' % manyruns_dirnames[idx])
        fpath = manyruns_paths[idx] + os.sep + 'dimreduce' + os.sep + 'dimreduce.z'
        if os.path.isfile(fpath):
            print('Exists already, loading: %s' % fpath)
            fcontents = joblib.load(fpath)  # just load file if it exists
            datasets[idx] = fcontents
        else:
            print('Dim. reduction on manyruns: %s' % manyruns_dirnames[idx])
            datasets[idx] = make_dimreduce_object(datasets[idx], umap_kwargs=umap_kwargs)
            save_dimreduce_object(datasets[idx], fpath)  # save to file (joblib)

    if add_control_data:
        print('adding control data...')
        total_spins_0 = datasets[0]['total_spins']
        num_runs_0 = datasets[0]['num_runs']

        # add control data into the dict of datasets
        control_X = generate_control_data(total_spins_0, num_runs_0)
        control_reducer = umap.UMAP(**umap_kwargs)
        control_reducer.fit(control_X)
        datasets[-1] = {
            'label': 'control (coin-flips)',
            'num_runs': num_runs_0,
            'total_spins': total_spins_0,
            'reducer': control_reducer,
            'embedding': control_reducer.transform(control_X),
            'energies': np.zeros((num_runs_0, 5)),
            'path': RUNS_FOLDER
        }

    # Step 3) vis data
    if vis_all:
        for idx in range(-1, len(manyruns_dirnames)):
            umap_plotly_express(datasets[idx])
