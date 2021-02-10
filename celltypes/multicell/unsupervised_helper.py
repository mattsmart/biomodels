import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
import joblib
import pandas as pd
import time

import plotly
import plotly.express as px
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
    
    and a separate dictionary 'algos' with keys for each algo (e.g. 'umap', 't-sne') 
        datasets[idx]['algos']['umap'] = {'reducer': umap.UMAP(**umap_kwargs)}
        datasets[idx]['algos']['umap']['reducer'].fit(X)
        datasets[idx]['algos']['umap']['reducer'].fit(X)
        datasets[idx]['algos']['umap']['embedding'] = datasets[idx]['reducer'].transform(X)
    
Here, each data subdict is pickled as a data_subdict pickle object
Regular location: 
    multicell_manyruns / gamma20.00e_10k / dimreduce / [files]
    files include dimreduce.pkl 
"""


# these set the defaults for modifications introduced in main
REDUCER_SEED = 100
REDUCER_COMPONENTS = 3
REDUCERS_TO_USE = ['umap', 'tsne', 'pca']
VALID_REDUCERS = ['umap', 'tsne', 'pca']
# see defaults: https://umap-learn.readthedocs.io/en/latest/api.html
UMAP_KWARGS = {
    'random_state': REDUCER_SEED,
    'n_components': REDUCER_COMPONENTS,
    'metric': 'euclidean',
    'init': 'spectral',
    'min_dist': 0.1,
    'spread': 1.0,
}
TSNE_KWARGS = {
    'random_state': REDUCER_SEED,
    'n_components': REDUCER_COMPONENTS,
    'metric': 'euclidean',
    'init': 'random',
    'perplexity': 30.0,
}
PCA_KWARGS = {
    'n_components': REDUCER_COMPONENTS,
}


def generate_control_data(total_spins, num_runs):
    X_01 = np.random.randint(2, size=(num_runs, total_spins))
    X = X_01 * 2 - 1
    return X


def make_dimreduce_object(data_subdict, flag_control=False, nsubsample=None, use_01=True,
                          umap_kwargs=UMAP_KWARGS,
                          pca_kwargs=PCA_KWARGS,
                          tsne_kwargs=TSNE_KWARGS):
    if flag_control:
        data_subdict['algos'] = {}
        X = data_subdict['data']
        if nsubsample is not None:
            X = X[0:nsubsample, :]
    else:
        manyruns_path = data_subdict['path']
        fpath_state = manyruns_path + os.sep + 'aggregate' + os.sep + 'X_aggregate.npz'
        fpath_energy = manyruns_path + os.sep + 'aggregate' + os.sep + 'X_energy.npz'
        fpath_pickle = manyruns_path + os.sep + 'multicell_template.pkl'
        print(fpath_state)
        X = np.load(fpath_state)['arr_0'].T  # umap wants transpose
        X_energies = np.load(fpath_energy)['arr_0'].T  # umap wants transpose (?)
        with open(fpath_pickle, 'rb') as pickle_file:
            multicell_template = pickle.load(pickle_file)  # unpickling multicell object

        if nsubsample is not None:
            X = X[0:nsubsample, :]
            X_energies = X_energies[0:nsubsample, :]

        # store data and metadata in datasets object
        num_runs, total_spins = X.shape
        print(X.shape)
        data_subdict['data'] = X
        data_subdict['index'] = list(range(num_runs))
        data_subdict['energies'] = X_energies
        data_subdict['num_runs'] = num_runs
        data_subdict['total_spins'] = total_spins
        data_subdict['multicell_template'] = multicell_template  # not needed? stored already
        data_subdict['algos'] = {}

    # binarization step needed for umap's binary metrics
    #  - convert +1, -1 to +1, 0
    if use_01:
        data_subdict['data'] = (1 + data_subdict['data']) / 2.0
        data_subdict['data'] = data_subdict['data'].astype(int)

    # perform dimension reduction
    for algo in REDUCERS_TO_USE:
        assert algo in VALID_REDUCERS
        data_subdict['algos'][algo] = {}

        t1 = time.time()
        if algo == 'umap':
            data_subdict['algos'][algo]['reducer'] = umap.UMAP(**umap_kwargs)
            data_subdict['algos'][algo]['reducer'].fit(X)
            embedding = data_subdict['algos'][algo]['reducer'].transform(X)
            data_subdict['algos'][algo]['embedding'] = embedding
        elif algo == 'pca':
            data_subdict['algos'][algo]['reducer'] = PCA(**pca_kwargs)
            embedding = data_subdict['algos'][algo]['reducer'].fit_transform(X)
            data_subdict['algos'][algo]['embedding'] = embedding
        else:
            assert algo == 'tsne'
            data_subdict['algos'][algo]['reducer'] = TSNE(**tsne_kwargs)
            embedding = data_subdict['algos'][algo]['reducer'].fit_transform(X)
            data_subdict['algos'][algo]['embedding'] = embedding
        print('Time to fit (%s): %.2f sec' % (algo, (time.time() - t1)))

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


def plotly_express_embedding(data_subdict, color_by_index=False, fmod='', show=False):
    """
    Supports 2D and 3D embeddings
    color_by_index: for troubleshooting, colors the points according to their array position
        if False (default), color by energy instead
    """

    num_runs = data_subdict['num_runs']
    label = data_subdict['label']
    dirpath = data_subdict['path'] + os.sep + 'dimreduce'

    if color_by_index:
        c = np.arange(num_runs)
        fmod += '_cIndex'
        clabel = 'index'
    else:
        c = data_subdict['energies'][:, 0]  # range(num_runs)
        clabel = 'energy'

    for key, algodict in data_subdict['algos'].items():
        algo = key
        embedding = algodict['embedding']

        n_components = embedding.shape[1]
        assert n_components in [2, 3]

        if n_components == 2:
            df = pd.DataFrame({'index': range(num_runs),
                               clabel: c,
                               'x': embedding[:, 0],
                               'y': embedding[:, 1]})

            fig = px.scatter(df, x='x', y='y',
                             color=clabel,
                             title='%s of %s dataset' % (algo, label))

        else:
            df = pd.DataFrame({'index': range(num_runs),
                               clabel: c,
                               'x': embedding[:, 0],
                               'y': embedding[:, 1],
                               'z': embedding[:, 2]})

            fig = px.scatter_3d(df, x='x', y='y', z='z',
                                color=clabel,
                                title='%s of %s dataset' % (algo, label))

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        fig.write_html(dirpath + os.sep + "%s_plotly_%s%s.html" % (algo, label, fmod))
        if show:
            fig.show()
    return


def pca_assess_dataset(data_subdict, fmod='', show=True):
    # see
    pca_full = PCA()
    pca_full.fit(data_subdict['data'])

    exp_var_cumul = np.cumsum(pca_full.explained_variance_ratio_)
    fig = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
    )

    dirpath = data_subdict['path'] + os.sep + 'dimreduce'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    fpath = dirpath + os.sep + "pca_cumvar%s.png" % (fmod)
    fig.write_image(fpath)
    print('pca cumvar saved to:\n%s' % fpath)
    if show:
        fig.show()
    return


if __name__ == '__main__':
    build_dimreduce_dicts = True
    add_control_data = False
    vis_all = True
    pca_assess = True
    subsample = True
    nsubsample = None  # None or an int

    # Step 0) which 'manyruns' dirs to work with
    gamma_list = [0.0, 0.05, 0.1, 0.2, 1.0, 2.0, 20.0]
    manyruns_dirnames = ['gamma%.2f_10k' % a for a in gamma_list]

    manyruns_paths = [RUNS_FOLDER + os.sep + 'multicell_manyruns' + os.sep + dirname
                      for dirname in manyruns_dirnames]

    # Step 1) umap (or other dim reduction) kwargs
    n_components = 3
    pca_kwargs = PCA_KWARGS.copy()
    umap_kwargs = UMAP_KWARGS.copy()
    tsne_kwargs = TSNE_KWARGS.copy()
    # modify pca settings
    pca_kwargs['n_components'] = n_components  # TODO don't need to spec 'live', can embed later?
    # modify umap settings
    umap_kwargs['n_components'] = n_components  # TODO don't need to spec 'live', can embed later?
    umap_kwargs['n_neighbors'] = 15
    umap_kwargs['min_dist'] = 0.99
    #umap_kwargs['spread'] = 1.0
    #umap_kwargs['metric'] = 'euclidean'
    # modify tsne settings
    tsne_kwargs['n_components'] = n_components  # TODO don't need to spec 'live', can embed later?
    tsne_kwargs['perplexity'] = 100

    # Modify filename suffix for dimreduce pkl and plots
    fmod = 'F=' + '+'.join(REDUCERS_TO_USE)
    fmod += '_dim%d_seed%d' % (umap_kwargs['n_components'], umap_kwargs['random_state'])
    if nsubsample is not None:
        fmod += '_nn%d' % nsubsample
    if 'umap' in REDUCERS_TO_USE:
        if umap_kwargs['metric'] != 'euclidean':
            fmod += '_%s' % umap_kwargs['metric']
        if umap_kwargs['init'] != 'spectral':
            fmod += '_%s' % umap_kwargs['init']
        if umap_kwargs['n_neighbors'] != 15:
            fmod += '_nbor%d' % umap_kwargs['n_neighbors']
        if umap_kwargs['min_dist'] != 0.1:
            fmod += '_dist%.2f' % umap_kwargs['min_dist']
        if umap_kwargs['spread'] != 1.0:
            fmod += '_spread%.2f' % umap_kwargs['spread']
    if 'tsne' in REDUCERS_TO_USE:
        if tsne_kwargs['perplexity'] != 30.0:
            fmod += '_perplex%.2f' % tsne_kwargs['perplexity']

    # Step 2) make/load data
    datasets = {i: {'label': manyruns_dirnames[i],
                    'path': manyruns_paths[i]}
                for i in range(len(manyruns_dirnames))}

    for idx in range(len(manyruns_dirnames)):
        fpath = manyruns_paths[idx] + os.sep + 'dimreduce' + os.sep + 'dimreduce%s.z' % fmod
        if os.path.isfile(fpath):
            print('Exists already, loading: %s' % fpath)
            fcontents = joblib.load(fpath)  # just load file if it exists
            datasets[idx] = fcontents
        else:
            print('Dim. reduction on manyruns: %s' % manyruns_dirnames[idx])
            datasets[idx] = make_dimreduce_object(
                datasets[idx], nsubsample=nsubsample, flag_control=False,
                umap_kwargs=umap_kwargs, tsne_kwargs=tsne_kwargs, pca_kwargs=pca_kwargs)
            save_dimreduce_object(datasets[idx], fpath)  # save to file (joblib)

    if add_control_data:
        print('adding control data...')
        total_spins_0 = datasets[0]['total_spins']
        num_runs_0 = datasets[0]['num_runs']

        # add control data into the dict of datasets
        control_X = generate_control_data(total_spins_0, num_runs_0)
        control_fpath = manyruns_paths[idx] + os.sep + 'dimreduce' + os.sep + 'dimreduce%s.z' % fmod

        datasets[-1] = {
            'data': control_X,
            'label': 'control (coin-flips)',
            'num_runs': num_runs_0,
            'total_spins': total_spins_0,
            'energies': np.zeros((num_runs_0, 5)),
            'path': RUNS_FOLDER
        }
        datasets[-1] = make_dimreduce_object(
            datasets[-1], flag_control=True, nsubsample=nsubsample,
            umap_kwargs=umap_kwargs, tsne_kwargs=tsne_kwargs, pca_kwargs=pca_kwargs)
        save_dimreduce_object(datasets[-1], control_fpath)  # save to file (joblib)

    # Step 3) vis data
    if vis_all:
        for idx in range(0, len(manyruns_dirnames)):
            plotly_express_embedding(datasets[idx], fmod=fmod, show=True)
            plotly_express_embedding(datasets[idx], fmod=fmod, color_by_index=True, show=False)

            if pca_assess and 'pca' in REDUCERS_TO_USE:
                pca_assess_dataset(datasets[idx], fmod=fmod, show=True)

        if add_control_data:
            plotly_express_embedding(datasets[-1], fmod=fmod)
            if pca_assess and 'pca' in REDUCERS_TO_USE:
                pca_assess_dataset(datasets[-1], fmod=fmod, show=True)
