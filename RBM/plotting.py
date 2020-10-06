import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from settings import DIR_OUTPUT

import matplotlib as mpl

#mpl.rcParams["mathtext.default"]
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = [r'\usepackage{bm}', r'\usepackage{amsmath}']
print(mpl.rcParams["text.usetex"])

sns.set()
sns.set(font_scale=1.2)  # use 1.2 for fig4 gen performance plots; see ipynb for other plots
sns.set_style("whitegrid")


def image_fancy(image, ax=None, show_labels=False):
    if ax is None:
        plt.figure()
        ax = plt.gca();
    im = ax.imshow(image, interpolation='none', vmin=0, vmax=1, aspect='equal', cmap='gray')

    # Minor ticks
    ax.set_xticks(np.arange(-.5, 28, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 28, 1), minor=True)

    if show_labels:
        # Major ticks
        ax.set_xticks(np.arange(0, 28, 1))
        ax.set_yticks(np.arange(0, 28, 1))
        # Labels for major ticks
        ax.set_xticklabels(np.arange(1, 29, 1))
        ax.set_yticklabels(np.arange(1, 29, 1))
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
    return ax


def image_fancy_wrapper(image, title='Example digit'):
    plt.figure()
    image_fancy(image, ax=plt.gca(), show_labels=False)
    plt.title(title)
    plt.show(); plt.close()


def plot_hopfield_generative_scores():
    palette = sns.cubehelix_palette(n_colors=10, start=.5, rot=-.75)  # see notebook for alt choices
    figsize = (4, 3)  # single column (4,3) estimate

    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    data_folder = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs' + os.sep + 'hopfield' + os.sep + 'original_AIS_TF_runs'
    out_dir = DIR_OUTPUT + os.sep + 'logZ' + os.sep + 'hopfield'

    beta_name = r'$\beta$'
    k_name = r'$k$'
    termA_name = r'$- \beta \langle H(s) \rangle$'
    score_name = r'$\frac{1}{M}\sum_a \ln p_{\boldsymbol{\theta}}(\mathbf{s}_a)$'  # or dfrac?
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

    plt.figure(figsize=figsize)
    ax = sns.lineplot(x=beta_name, y=score_name, hue=k_name, marker='o', markers=True, dashes=False, data=df1,
                      legend='full', palette=palette)
    # plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
    # plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
    plt.legend(fontsize='x-small', loc='lower left', ncol=2)
    ax.tick_params(axis='both', which='major', pad=-1)

    # For gridlines soften
    ax.tick_params(grid_alpha=0.5)

    plt.savefig(out_dir + os.sep + 'kvary_scores.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=figsize)
    ax = sns.lineplot(x=beta_name, y=termA_name, hue=k_name, marker='o', markers=True, dashes=False, data=df1,
                      legend='full', palette=palette)
    plt.savefig(out_dir + os.sep + 'kvary_termA.pdf')
    plt.show()
    plt.close()

    plt.figure(figsize=figsize)
    ax = sns.lineplot(x=beta_name, y=LogZ_name, hue=k_name, marker='o', markers=True, dashes=False, data=df1,
                      legend='full', palette=palette)
    plt.savefig(out_dir + os.sep + 'kvary_logZ.pdf')
    plt.show()
    plt.close()

    return


def compare_generative_scores(plotting_dict, out_dir):

    flatui_a = ["#3498db", "#9b59b6", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    flatui_b = ['#1f77b4', 'mediumpurple', "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    flatui_c = ["#3498db", 'mediumpurple', "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    sns.set_palette(flatui_c)

    figsize = (4,3)
    epoch_name = r'$\textrm{epoch}$'
    category_name = r'$Initial_weights$'
    score_name = r'$\frac{1}{M}\sum_a \ln p_{\boldsymbol{\theta}}(\mathbf{s}_a)$'  # or dfrac?
    termA_name = r'$- \beta \langle H(s) \rangle$'
    LogZ_name = r'$\ln \ Z$'

    # need to crate pandas object to pass to sns lineplot https://seaborn.pydata.org/generated/seaborn.lineplot.html
    # example: replace event column elements with 'k' https://github.com/mwaskom/seaborn-data/blob/master/fmri.csv
    df1 = pd.DataFrame({epoch_name: [], category_name: [], score_name: [], termA_name: [], LogZ_name: []})

    for k, v in plotting_dict.items():
        for idx, epoch in enumerate(v['epochs']):
            datarow = [{epoch_name: epoch,
                        category_name: v['category'],
                        score_name: v['score'][idx],
                        termA_name: v['termA'][idx],
                        LogZ_name: v['logZ'][idx]}]
            df1 = df1.append(datarow, ignore_index=True)

    plt.figure(figsize=figsize)
    ax = sns.lineplot(x=epoch_name, y=score_name, hue=category_name, dashes=False, legend='full', data=df1)
    #ax = sns.lineplot(x=epoch_name, y=score_name, hue=category_name, marker='o', markers=True, dashes=False, data=df1,
    #                  legend='full')
    plt.ylim(-500,-50)
    #plt.ylim(-250,-50)
    plt.xlim(0, 50)
    ax.legend().texts[0].set_text(r'$\textrm{Initial RBM weights } \mathbf{W}_\textrm{init}$')
    #plt.setp(ax.get_legend().get_texts(), fontsize='small')

    # soften grid
    ax.tick_params(grid_alpha=0.5)

    plt.savefig(out_dir + os.sep + 'scores.pdf', bbox_inches='tight')
    plt.show(); plt.close()

    plt.figure(figsize=figsize)
    ax = sns.lineplot(x=epoch_name, y=termA_name, hue=category_name, marker='o', markers=True, dashes=False, data=df1,
                      legend='full')
    plt.savefig(out_dir + os.sep + 'termA.pdf')
    plt.show(); plt.close()

    plt.figure(figsize=figsize)
    ax = sns.lineplot(x=epoch_name, y=LogZ_name, hue=category_name, marker='o', markers=True, dashes=False, data=df1,
                      legend='full')
    plt.savefig(out_dir + os.sep + 'logZ.pdf')
    plt.show(); plt.close()

    return


def compare_generative_scores_sep(plotting_dict, out_dir):
    figsize = (4,3)
    epoch_name = r'$\textrm{epoch}$'
    category_name = r'$Initial_weights$'
    score_name = r'$\frac{1}{M}\sum_a \ln p_{\boldsymbol{\theta}}(\mathbf{s}_a)$'  # or dfrac?
    termA_name = r'$- \beta \langle H(s) \rangle$'
    LogZ_name = r'$\ln \ Z$'

    kwdict = {r'$\textrm{Hopfield mapping}$':
                  {'c': '#1f77b4', 'z':10},
              r'$W_{i\mu}\sim\mathcal{N}(0,0.01)$':
                  {'c': 'mediumpurple', 'z':3},
              r'$\textrm{From Hopfield mapping}$ + biases':
                  {'c': '#1f77b4', 'z': 3, 'linestyle': '--'},
              r'$W_{i\mu}\sim\mathcal{N}(0,0.01)$ + biases':
                  {'c': 'mediumpurple', 'z': 2, 'linestyle': '--'}
              }

    plt.figure(figsize=figsize)
    for k, v in plotting_dict.items():
        print(k)
        print(kwdict[v['category']])
        print(kwdict[v['category']]['c'])
        print(kwdict[v['category']]['z'])
        plt.plot(v['epochs'], v['score'], label=v['title'], alpha=0.8,
                 color=kwdict[v['category']]['c'], zorder=kwdict[v['category']]['z'])
    plt.xlabel(epoch_name); plt.ylabel(score_name)
    plt.ylim(-500,-50)  # plt.ylim(-250,-50)
    plt.xlim(0, 50)
    plt.savefig(out_dir + os.sep + 'scores_sep.pdf', bbox_inches='tight')
    plt.show(); plt.close()

    return


def plot_classify_dict(plotting_dict, k_to_plot, out_dir, figsize=(3.4, 3)):
    epoch_name = r'$\textrm{epoch}$'
    category_name = r'$Initial_weights$'
    error_name = r'$\textrm{Test error (\%)}$'  # or dfrac?
    k_name = r'$k$'

    # plt.figure()
    plt.figure(figsize=figsize); ax = plt.gca()
    for a in ['norm', 'hopf']:

        for k in k_to_plot:

            klabel = r'$k=%d$' % k
            vc = plotting_dict[a][klabel]
            plt.errorbar(vc['x'], vc['y_mean'], yerr=vc['y_err'],
                         marker=vc['marker'],
                         label=plotting_dict[a]['label'] + klabel,
                         color=plotting_dict[a]['c'],
                         linestyle=plotting_dict[a]['ls'],
                         alpha=0.8,
                         markersize=5,
                         linewidth=0.5,
                         markeredgecolor='k',
                         markeredgewidth=0.5,
                         ecolor='k',
                         elinewidth=1,
                         capsize=2,
                         zorder=plotting_dict[a]['z'])

            if k == 10:
                print(k, a, vc['y_mean'])
                print()

            plt.fill_between(
                vc['x'],
                vc['y_mean'] - vc['y_err'][0, :],
                vc['y_mean'] + vc['y_err'][1, :],
                color=plotting_dict[a]['c'],  # color='gray',
                alpha=0.2)

            #print(vc['y_err'][1, :] - vc['y_mean'], )

    plt.xlabel(epoch_name); plt.ylabel(error_name)
    plt.ylim(1.7, 11.9)  # plt.ylim(1.5, 9.9)
    plt.xlim(-1.5, 51.5)

    # make legend
    #plt.legend()
    #ax.legend().texts[0].set_text(r'$\textrm{Initial RBM weights } \mathbf{W}_\textrm{init}$')
    #ax.legend().texts[0].set_text(r'$\textrm{Number of sub-patterns} k: \mathbf{W}_\textrm{init}$')
    #plt.setp(ax.get_legend().get_texts(), fontsize='small')

    # tick settings and grid
    plt.xticks(np.arange(0, 51, 10.0))
    # For gridlines soften AND add ticks
    """plt.gca().tick_params(bottom=True, left=True, right=True,
                          direction='in',
                          color='#cccccc')"""
    # For gridlines soften
    ax.tick_params(grid_alpha=0.5)

    plt.savefig(out_dir + os.sep + 'errors2_expt.pdf', bbox_inches='tight')
    plt.show(); plt.close()

    return


if __name__ == '__main__':
    plot_hopfield_k_generative = True
    plot_compare_generative = False
    plot_compare_classify = False

    if plot_hopfield_k_generative:
        plot_hopfield_generative_scores()

    if plot_compare_generative:
        scores_to_compare = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs' + os.sep + 'scores_to_compare'
        #compare_dir = scores_to_compare + os.sep + 'aug10_hopfield_vs_normal_10p_100batch_1e-4eta'
        compare_dir = scores_to_compare + os.sep + 'sept26_hopfield_vs_normal_50p_100batch_etaVary5to1normalOnly_50'

        def get_category_info(plot_key, use_fields):
            post = ''
            if use_fields:
                post = ' + biases'

            if plot_key[0:3] == 'hop':
                val = r'$\textrm{Hopfield mapping}$' + post
            elif plot_key[0:3] == 'nor':
                #val = r'$(\mathbf{W}_\mathrm{init})_{\it{i}\mu}\sim\mathcal{N}(0,0.01)$' + post
                val = r'$W_{i\mu}\sim\mathcal{N}(0,0.01)$' + post
            else:
                assert 1 == 2
            return val

        plotting_dict = {}
        onlynpz = [f for f in os.listdir(compare_dir) if
                   (os.path.isfile(os.path.join(compare_dir, f)) and f[-4:] == '.npz')]
        for f in onlynpz:
            f_info = f.split('_')
            plot_key = f_info[-1][:-4]
            use_fields = bool(int(f_info[2][0]))
            plot_key += f_info[2][0]

            category = get_category_info(plot_key, use_fields)

            dataobj = np.load(compare_dir + os.sep + f)
            plotting_dict[plot_key] = \
                {'epochs': dataobj['epochs'],
                 'termA': dataobj['termA'],
                 'logZ': dataobj['logZ'],
                 'score': dataobj['score'],
                 'category': category,
                 'title': plot_key}

        compare_generative_scores_sep(plotting_dict, compare_dir)
        compare_generative_scores(plotting_dict, compare_dir)

    if plot_compare_classify:

        scores_to_compare = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs' + os.sep + 'scores_to_compare'
        compare_dir = scores_to_compare + os.sep + 'sept27_classify_poe'

        K_TO_KEEP = [10, 20, 100]  # [1,10,100,500]

        k_to_marker = {1: 'o',
                       10: 'o',
                       20: 's',
                       100: '^',
                       200: '*',
                       250: '*',
                       300: '*',
                       500: '*'}

        onlynpz = [f for f in os.listdir(compare_dir) if
                   (os.path.isfile(os.path.join(compare_dir, f)) and f[-4:] == '.npz')]

        # prepare plotting dict
        plotting_dict = {a: {'curves': {}, 'num_runs': 0}
                         for a in ['norm', 'hopf']}
        plotting_dict['norm'].update(
            [('label', r'$W_{i\mu}\sim\mathcal{N}(0,0.01)$'),
             ('c', '#a77ad1'),  # mediumpurple is #9370DB (alt is #a77ad1)
             ('ls', '-.'),
             ('z', 5)
             ])
        plotting_dict['hopf'].update(
            [('label', r'$\textrm{From Hopfield mapping}$'),
             ('c', '#2580d4'),  # "#3498db" (alt is #2580d4) or #1f77b4
             ('ls', '--'),
             ('z', 10)
             ])

        # STEP 1: loop over the npz and gather data
        for f in onlynpz:
            f_info = f.split('_')
            plot_key = f_info[-1][:-4]  # 'e.g. 'hopf', 'norm'
            plot_code = plot_key[0:4]
            assert plot_code in ['hopf', 'norm']

            # make sure use_fields is False
            use_fields = bool(int(f_info[2][0]))
            # plot_key += f_info[2][0]
            assert not use_fields  # currently not supported

            # load the data
            dataobj = np.load(compare_dir + os.sep + f)
            accs = dataobj['accs_epoch_by_k']
            errors_epoch_by_k = 100 * (1 - accs)
            k_range = dataobj['k_range']
            epochs = dataobj['epochs']

            print(plot_code, 'entering kloop LOOP 1')
            for k_idx, k in enumerate(k_range):
                current_run = plotting_dict[plot_code]['num_runs']
                if k in K_TO_KEEP:
                    curve_label = r'$k=%d$' % k
                    yvals = errors_epoch_by_k[:, k_idx]
                    if curve_label in plotting_dict[plot_code].keys():
                        assert (epochs == plotting_dict[plot_code][curve_label]['x']).all()
                        plotting_dict[plot_code][curve_label]['y_sum'] += yvals
                        plotting_dict[plot_code][curve_label]['runs'][current_run] = yvals
                    else:
                        print(plot_code, curve_label, current_run)
                        assert current_run == 0
                        plotting_dict[plot_code][curve_label] = {
                            'x': epochs,
                            'y_sum': np.copy(yvals),
                            'marker': k_to_marker[k],
                            'runs': {0: yvals}}
            plotting_dict[plot_code]['num_runs'] += 1

        print("\n\nLOOP 1 keys")
        print(plotting_dict['hopf'].keys())
        print(plotting_dict['norm'].keys())

        # STEP 2: compute curve means and error bars
        for plot_label in ['hopf', 'norm']:
            num_runs = plotting_dict[plot_code]['num_runs']
            print(num_runs, num_runs, num_runs, num_runs, num_runs)
            for k in K_TO_KEEP:
                curve_label = r'$k=%d$' % k
                ysum = plotting_dict[plot_label][curve_label]['y_sum']
                plotting_dict[plot_label][curve_label]['y_mean'] = ysum / num_runs

                num_runs = plotting_dict[plot_label]['num_runs']
                val_arr = np.array([plotting_dict[plot_label][curve_label]['runs'][r]
                                    for r in range(num_runs)])
                #
                print("num_runs", num_runs)
                #print("val_arr.shape", val_arr.shape)
                # val_arr = np.array(plotting_dict[plot_label][curve_label]['runs'])

                # y_SEM = sp.stats.sem(
                #    np.array(plotting_dict[plot_label][curve_label]['yvals_list']), axis=0)
                # y_STD = np.std(
                #    np.array(plotting_dict[plot_label][curve_label]['yvals_list']), axis=0)
                # y_err = y_STD

                # print('val arr col0')
                # print(val_arr[:, 0])

                y_upper = np.max(val_arr, axis=0) - plotting_dict[plot_label][curve_label]['y_mean']
                y_lower = plotting_dict[plot_label][curve_label]['y_mean'] - np.min(val_arr, axis=0)
                y_err = np.concatenate(([y_lower], [y_upper]))

                # print("y_mean")
                # print(plotting_dict[plot_label][curve_label]['y_mean'])
                # print("y_upper.shape", y_upper.shape)
                # print("y_err.shape", y_err.shape)
                # print(y_err[0,:])
                # print(y_err[1,:])
                # print(val_arr[:,0])
                plotting_dict[plot_label][curve_label]['y_err'] = y_err

                print(plot_label, k, '...')
                print(plotting_dict[plot_label][curve_label]['y_mean'])

        #print("\n\nLOOP 2 keys")
        #print(plotting_dict['hopf'].keys())

        plot_classify_dict(plotting_dict, K_TO_KEEP, compare_dir)
