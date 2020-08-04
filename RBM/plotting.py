import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns; sns.set()

from settings import DIR_OUTPUT


def image_fancy(image, ax=None, show_labels=False):
    if ax is None:
        plt.figure()
        ax = plt.gca();
    im = ax.imshow(image, interpolation='none', vmin=0, vmax=1, aspect='equal', cmap='gray');

    # Minor ticks
    ax.set_xticks(np.arange(-.5, 28, 1), minor=True);
    ax.set_yticks(np.arange(-.5, 28, 1), minor=True);

    if show_labels:
        # Major ticks
        ax.set_xticks(np.arange(0, 28, 1));
        ax.set_yticks(np.arange(0, 28, 1));
        # Labels for major ticks
        ax.set_xticklabels(np.arange(1, 29, 1));
        ax.set_yticklabels(np.arange(1, 29, 1));
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
    return ax


def plot_hopfield_generative_scores():
    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    data_folder = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs' + os.sep + 'hopfield'

    beta_name = r'$\beta$'
    k_name = r'$k$'
    score_name = r'$\langle\ln \ p(x)\rangle$'
    termA_name = r'$- \beta \langle H(s) \rangle$'
    LogZ_name = r'$\ln \ Z$'

    # need to crate pandas object to pass to sns lineplot https://seaborn.pydata.org/generated/seaborn.lineplot.html
    # example: replace event column elements with 'k' https://github.com/mwaskom/seaborn-data/blob/master/fmri.csv
    df1 = pd.DataFrame({beta_name: [], k_name: [], score_name: [], termA_name: [], LogZ_name: []})
    for k in k_list:
        npzpath = data_folder + os.sep + 'objective_%dpatterns_200steps.npz' % k
        dataobj = np.load(npzpath)
        runs = dataobj['termA'].shape[0]
        """
        datarows = [
            [beta, k, dataobj['score'][idx, b], dataobj['termA'][idx, b], dataobj['logZ'][idx, b]]
            for b, beta in enumerate(dataobj['beta'])
            for idx in range(runs)]
        df1 = df1.append(datarows, ignore_index=True)
        print (df1)
        """
        for b, beta in enumerate(dataobj['beta']):
            for idx in range(runs):
                datarow = [{beta_name: beta, k_name: k,
                            score_name: dataobj['score'][idx, b],
                            termA_name: dataobj['termA'][idx, b],
                            LogZ_name: dataobj['logZ'][idx, b]}]
                df1 = df1.append(datarow, ignore_index=True)
    out_dir = DIR_OUTPUT + os.sep + 'logZ' + os.sep + 'hopfield'

    plt.figure()
    ax = sns.lineplot(x=beta_name, y=score_name, hue=k_name, marker='o', markers=True, dashes=False, data=df1,
                      legend='full')
    plt.savefig(out_dir + os.sep + 'kvary_scores.pdf')
    plt.show()
    plt.close()

    plt.figure()
    ax = sns.lineplot(x=beta_name, y=termA_name, hue=k_name, marker='o', markers=True, dashes=False, data=df1,
                      legend='full')
    plt.savefig(out_dir + os.sep + 'kvary_termA.pdf')
    plt.show()
    plt.close()

    plt.figure()
    ax = sns.lineplot(x=beta_name, y=LogZ_name, hue=k_name, marker='o', markers=True, dashes=False, data=df1,
                      legend='full')
    plt.savefig(out_dir + os.sep + 'kvary_logZ.pdf')
    plt.show()
    plt.close()

    return


if __name__ == '__main__':
    plot_hopfield_generative_scores()
