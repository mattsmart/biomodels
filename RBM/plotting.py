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


def compare_generative_scores(plotting_dict, out_dir, assume_epochs=False):

    flatui_a = ["#3498db", "#9b59b6", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    flatui_b = ['#1f77b4', 'mediumpurple', "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    flatui_c = ["#3498db", 'mediumpurple', "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    flatui_d = ["#3498db", 'mediumpurple', "#a1e4cc", "#ff9aa2", "#34495e", "#2ecc71"]
    flatui_d_rearrange_4 = ["#a1e4cc", "#3498db", 'mediumpurple', "#ff9aa2"]
    flatui_d_rearrange_3 = ["#a1e4cc", "#3498db", "#ff9aa2"]
    sns.set_palette(flatui_d_rearrange_4)  # was flatui_d, use if only hopfield and normal

    figsize = (4,3)  # default: 4,3  -- inset: 2, 1.2
    if assume_epochs:
        x_name = r'$\textrm{epoch}$'
    else:
        x_name = r'$\textrm{iteration}$'
    category_name = r'$Initial_weights$'
    score_name = r'$\frac{1}{M}\sum_a \ln p_{\boldsymbol{\theta}}(\mathbf{s}_a)$'  # or dfrac?
    termA_name = r'$- \beta \langle H(s) \rangle$'
    LogZ_name = r'$\ln \ Z$'
    zorder_name = 'zo'

    # need to crate pandas object to pass to sns lineplot https://seaborn.pydata.org/generated/seaborn.lineplot.html
    # example: replace event column elements with 'k' https://github.com/mwaskom/seaborn-data/blob/master/fmri.csv
    df1 = pd.DataFrame({x_name: [], category_name: [], score_name: [], termA_name: [], LogZ_name: [], zorder_name: []})

    for k, v in plotting_dict.items():
        for idx, timepoint in enumerate(v['timepoints']):
            datarow = [{x_name: timepoint,
                        category_name: v['category'],
                        score_name: v['score'][idx],
                        termA_name: v['termA'][idx],
                        LogZ_name: v['logZ'][idx],
                        zorder_name: v['zo']}]
            df1 = df1.append(datarow, ignore_index=True)

    plt.figure(figsize=figsize)
    # EXTRA OPTIONS: legend False or 'full', marker='o'
    ax = sns.lineplot(x=x_name, y=score_name, hue=category_name, dashes=False, data=df1, legend=False, marker='o') #legend='full' or False   , marker='o'
    #ax = sns.lineplot(x=epoch_name, y=score_name, hue=category_name, marker='o', markers=True, dashes=False, data=df1,
    #                  legend='full')

    plt.ylim(-500, -150)   # plt.ylim(-500,-50)  plt.ylim(-450,-80) (-320, -70) [(-500, -150) for inset]
    plt.xlim(-2, 120)  # plt.xlim(-0.5, 50)

    #ax.legend().texts[0].set_text(r'$\textrm{Initial RBM weights } \mathbf{W}_\textrm{init}$')
    #plt.setp(ax.get_legend().get_texts(), fontsize='small')

    fancy_axis = False
    if fancy_axis:

        # Add x ticks manually
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator( [0, 60, 120] ))
        ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator( [30, 90] ))
        ax.set_xticklabels([r'$0$', r'$1$', r'$2$'])

        # Add y ticks manually
        ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([-400, -200]))
        ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([-300, -100]))
        ax.set_yticklabels([r'$-400$', r'$-200$'])

        # Include xticks and yticks pointing in
        ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on', bottom='on', color='#cacaca')
        ax.xaxis.set_tick_params(which='minor', size=3.5, width=1, direction='in', top='on', bottom='on', color='#cacaca')
        ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on', left='on', color='#cacaca')
        ax.yaxis.set_tick_params(which='minor', size=3.5, width=1, direction='in', right='on', left='on', color='#cacaca')


    # soften grid
    ax.tick_params(grid_alpha=0.5)
    plt.savefig(out_dir + os.sep + 'scores.pdf', bbox_inches='tight')
    plt.show(); plt.close()

    plt.figure(figsize=figsize)
    ax = sns.lineplot(x=x_name, y=termA_name, hue=category_name, marker='o', markers=True, dashes=False, data=df1,
                      legend='full')
    plt.savefig(out_dir + os.sep + 'termA.pdf')
    plt.show(); plt.close()

    plt.figure(figsize=figsize)
    ax = sns.lineplot(x=x_name, y=LogZ_name, hue=category_name, marker='o', markers=True, dashes=False, data=df1,
                      legend='full')
    plt.savefig(out_dir + os.sep + 'logZ.pdf')
    plt.show(); plt.close()

    return


def compare_generative_scores_sep(plotting_dict, out_dir, assume_epochs=False):

    figsize = (4,3)
    if assume_epochs:
        x_name = r'$\textrm{epoch}$'
    else:
        x_name = r'$\textrm{iteration}$'
    category_name = r'$Initial_weights$'
    score_name = r'$\frac{1}{M}\sum_a \ln p_{\boldsymbol{\theta}}(\mathbf{s}_a)$'  # or dfrac?
    termA_name = r'$- \beta \langle H(s) \rangle$'
    LogZ_name = r'$\ln \ Z$'

    #'#b5ead7' original mint hebbian
    # old label: r'$\textrm{Hopfield mapping (Hebbian)}$'
    kwdict = {r'$\textrm{Hopfield (projection)}$':
                  {'c': '#3498db', 'z':10},
              r'$W_{i\mu}\sim\mathcal{N}(0,0.01)$':
                  {'c': 'mediumpurple', 'z':3},
              r'$\textrm{Hopfield (projection)}$ + biases':
                  {'c': '#3498db', 'z': 3, 'linestyle': '--'},
              r'$W_{i\mu}\sim\mathcal{N}(0,0.01)$ + biases':
                  {'c': 'mediumpurple', 'z': 2, 'linestyle': '--'},
              r'$\textrm{Hopfield (Hebbian)}$':
                  {'c': '#a1e4cc', 'z': 12},
              r'$\textrm{PCA}$':
                  {'c': '#ff9aa2', 'z': 2},
              r'$\textrm{Hopfield (Hebbian)}$ + biases':
                  {'c': '#a1e4cc', 'z': 12, 'linestyle': '--'},
              r'$\textrm{PCA weights}$ + biases':
                  {'c': '#ff9aa2', 'z': 2, 'linestyle': '--'},
              }

    plt.figure(figsize=figsize)
    for k, v in plotting_dict.items():
        print(k)
        print(kwdict[v['category']])
        print(kwdict[v['category']]['c'])
        print(kwdict[v['category']]['z'])
        print(v.keys())
        plt.plot(v['timepoints'], v['score'], label=v['title'], alpha=0.6, # TODO was 0.8, and no marker
                 color=kwdict[v['category']]['c'], zorder=kwdict[v['category']]['z'], marker='o')
    plt.xlabel(x_name); plt.ylabel(score_name)
    plt.ylim(-500, -150)   # plt.ylim(-500,-50)  plt.ylim(-450,-80) (-320, -70)
    #plt.xlim(-0.5, 50)
    plt.gca().tick_params(grid_alpha=0.5)

    plt.savefig(out_dir + os.sep + 'scores_sep.pdf', bbox_inches='tight')
    plt.show(); plt.close()

    return


def plot_classify_dict_BACKUP(plotting_dict, k_to_plot, out_dir, figsize=(3.55, 3.15), plot_codes=('nor', 'hop'),
                       xlim=None, legend=False, fmod=''):
    # Figsize (pre - Nov 14): 3.4, 3;  experimental: 3.55, 3.15 (for 50 epochs)


    epoch_name = r'$\textrm{epoch}$'
    category_name = r'$Initial_weights$'
    error_name = r'$\textrm{Test error (\%)}$'  # or dfrac?
    k_name = r'$k$'

    # plt.figure()
    plt.figure(figsize=figsize); ax = plt.gca()
    for a in plot_codes:

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
    plt.ylim(2.4, 11.9)  # PRE-NOV: ylim(1.7, 11.9)
    if xlim is None:
        plt.xlim(-1.5, 51.5)
    else:
        plt.xlim(xlim[0], xlim[1])

    # make legend
    if legend:
        plt.legend()
        ax.legend().texts[0].set_text(r'$\textrm{Initial RBM weights } \mathbf{W}_\textrm{init}$')
        ax.legend().texts[0].set_text(r'$\textrm{Number of sub-patterns} k: \mathbf{W}_\textrm{init}$')
        plt.setp(ax.get_legend().get_texts(), fontsize='small')

    # tick settings and grid
    plt.xticks(np.arange(0, 51, 10.0))
    # For gridlines soften AND add ticks
    """plt.gca().tick_params(bottom=True, left=True, right=True,
                          direction='in',
                          color='#cccccc')"""
    # For gridlines soften
    ax.tick_params(grid_alpha=0.5)

    plt.savefig(out_dir + os.sep + 'classify%s.pdf' % fmod, bbox_inches='tight')
    plt.show(); plt.close()

    return


def plot_classify_dict(plotting_dict, k_to_plot, out_dir, figsize=(3.55, 3.15), plot_codes=('nor', 'hop'),
                       xlim=None, legend=False, fmod=''):
    # Figsize (pre - Nov 14): 3.4, 3;  experimental: 3.55, 3.15 (for 50 epochs)
    manual_x = False
    manual_y = True

    epoch_name = r'$\textrm{epoch}$'
    error_name = r'$\textrm{Test error (\%)}$'  # or dfrac?

    # plt.figure()
    plt.figure(figsize=figsize); ax = plt.gca()
    for a in plot_codes:

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

    # make legend
    if legend:
        plt.legend()
        ax.legend().texts[0].set_text(r'$\textrm{Initial RBM weights } \mathbf{W}_\textrm{init}$')
        ax.legend().texts[0].set_text(r'$\textrm{Number of sub-patterns} k: \mathbf{W}_\textrm{init}$')
        plt.setp(ax.get_legend().get_texts(), fontsize='small')

    # AXIS PREP
    ax.tick_params(grid_alpha=0.5)  # For gridlines soften
    ylim = (2.7, 12.9)  # PRE-NOV: ylim(1.7, 11.9)
    plt.ylim()
    if xlim is None:
        xlim = (-1.5, 51.5)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])


    if manual_x:
        ax.grid(False)

        # tick settings and grid
        plt.xticks(np.arange(0, xlim[1], 10.0))  # was 51
        # For gridlines soften

        # Add x ticks manually
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator( [0, 20, 40, 60, 80, 100] ))
        ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator( [10, 30, 50, 70, 90] ))
        #ax.set_xticklabels([r'$0$', r'$1$', r'$2$'])

        # Add y ticks manually
        #ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([-400, -200]))
        #ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([-300, -100]))
        #ax.set_yticklabels([r'$-400$', r'$-200$'])

        # Include xticks and yticks pointing in
        ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on', bottom='on', color='#cacaca')
        ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on', bottom='on', color='#cacaca')
    else:
        plt.xticks(np.arange(0, xlim[1], 10.0))  # was 51

    if manual_y:
        if xlim[1] >= 100:
            ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(list(range(3,12))))
        else:
            ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(list(range(3, 12, 2))))
        ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator( [3, 5, 7, 9, 11] ))
        #ax.yaxis.set_tick_params(which='major', size=10, width=1, direction='in', right='on', left='on', color='#cacaca')
        #ax.yaxis.set_tick_params(which='minor', size=4, width=1, direction='in', right='on', left='on', color='#cacaca')

    plt.savefig(out_dir + os.sep + 'classify%s.pdf' % fmod, bbox_inches='tight')
    plt.show(); plt.close()

    return

if __name__ == '__main__':
    plot_hopfield_k_generative = False
    plot_compare_generative = False
    plot_compare_classify = True

    if plot_hopfield_k_generative:
        plot_hopfield_generative_scores()

    if plot_compare_generative:
        scores_to_compare = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs' + os.sep + 'scores_to_compare'
        #compare_dir = scores_to_compare + os.sep + 'sept24_hopfield_vs_normal_10p_100batch_1e-4eta'
        #compare_dir = scores_to_compare + os.sep + 'sept26_hopfield_vs_normal_50p_100batch_etaVary5to1normalOnly_50'

        compare_dir = scores_to_compare + os.sep + 'nov13_fig4a_NewInit_FewEpochs_1000batch'
        #compare_dir = scores_to_compare + os.sep + 'nov12_fig4b_NewInit_RegEpochs'

        assume_epochs = False  # assumes the timeseries is one point per epoch (old mode)
        if assume_epochs:
            timepoints_key = 'epochs'
        else:
            timepoints_key = 'iterations'

        def get_category_info(plot_key, use_fields):
            post = ''
            if use_fields:
                post = ' + biases'

            if plot_key[0:3] == 'hop':
                #val = r'$\textrm{Hopfield mapping}$' + post
                val = r'$\textrm{Hopfield (projection)}$' + post
                zorder = 10
            elif plot_key[0:3] == 'nor':
                #val = r'$(\mathbf{W}_\mathrm{init})_{\it{i}\mu}\sim\mathcal{N}(0,0.01)$' + post
                val = r'$W_{i\mu}\sim\mathcal{N}(0,0.01)$' + post
                zorder = 1
            elif plot_key[0:3] == 'heb':
                val = r'$\textrm{Hopfield (Hebbian)}$' + post
                zorder = 4
            elif plot_key[0:3] == 'pca':
                val = r'$\textrm{PCA}$' + post
                zorder = 2
            else:
                assert 1 == 2
            return val, zorder

        plotting_dict = {}
        onlynpz = [f for f in os.listdir(compare_dir) if
                   (os.path.isfile(os.path.join(compare_dir, f)) and f[-4:] == '.npz')]
        for f in onlynpz:
            f_info = f.split('_')
            plot_key = f_info[-1][:-4]
            use_fields = bool(int(f_info[2][0]))
            plot_key += f_info[2][0]

            category, zorder = get_category_info(plot_key, use_fields)

            dataobj = np.load(compare_dir + os.sep + f)
            plotting_dict[plot_key] = \
                {'timepoints': dataobj[timepoints_key],
                 'termA': dataobj['termA'],
                 'logZ': dataobj['logZ'],
                 'score': dataobj['score'],
                 'category': category,
                 'zo': zorder,  # TODO
                 'title': plot_key}

        #compare_generative_scores_sep(plotting_dict, compare_dir, assume_epochs=assume_epochs)
        compare_generative_scores(plotting_dict, compare_dir, assume_epochs=assume_epochs)

    if plot_compare_classify:

        scores_to_compare = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs' + os.sep + 'scores_to_compare'
        #compare_dir = scores_to_compare + os.sep + 'sept27_classify_poe'
        #compare_dir = scores_to_compare + os.sep + 'nov15_classify_poe_newinit_all25points'
        compare_dir = scores_to_compare + os.sep + 'nov15_classify_poe_newinit_all25points_noHebb'

        #xlim = None # (-1.5, 101.5)  # Options: (-1.5, 100) or None
        xlim = (-1.5, 50.7)  # Options: (-1.5, 100) or None
        legend = False
        separate_plots = False
        separate_by_init = False  # if False and separate_plots, then sep by k value
        figsize = (3.55, 3.15)  # set to None for def (3.55, 3.15), big is (7, 3.15) or (7, 6.3) (SI)

        PLOT_CODES_MAX = ['hop', 'nor', 'PCA', 'heb']
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
                         for a in PLOT_CODES_MAX}
        plotting_dict['nor'].update(
            [('label', r'$W_{i\mu}\sim\mathcal{N}(0,0.01)$'),
             ('c', '#a77ad1'),  # mediumpurple is #9370DB (alt is #a77ad1)
             ('ls', '-.'),
             ('z', 15)
             ])
        plotting_dict['hop'].update(
            [('label', r'$\textrm{Hopfield (projection)}$'),
             ('c', '#2580d4'),  # "#3498db" (alt is #2580d4) or #1f77b4
             ('ls', '-'),
             ('z', 10)
             ])
        plotting_dict['heb'].update(
            [('label', r'$\textrm{Hopfield (Hebbian)}$'),
             ('c', '#a1e4cc'),
             ('ls', ':'),      # TODO different from hopfield
             ('z', 8)
             ])
        plotting_dict['PCA'].update(
            [('label', r'$\textrm{PCA}$'),
             ('c', '#ff9aa2'),
             ('ls', '--'),      # TODO different from hopfield
             ('z', 6)
             ])

        # STEP 1: loop over the npz and gather data
        for f in onlynpz:
            f_info = f.split('_')
            plot_key = f_info[-1][:-3]  # 'e.g. 'hop', 'nor', 'PCA', 'heb'
            plot_code = plot_key[0:3]
            assert plot_code in PLOT_CODES_MAX

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
        PLOT_CODES_PRESENT = []
        for plot_label in PLOT_CODES_MAX:
            print(plot_label, plotting_dict[plot_label].keys(), "(num_runs = %d)" % plotting_dict[plot_label]['num_runs'])
            if plotting_dict[plot_label]['num_runs'] > 0:
                PLOT_CODES_PRESENT.append(plot_label)

        # STEP 2: compute curve means and error bars
        for plot_label in ['hop', 'nor', 'PCA', 'heb']:
            num_runs = plotting_dict[plot_label]['num_runs']
            print("Step 2 for:", plot_label, "(num_runs = %d)" % num_runs)
            if num_runs > 0:
                for k in K_TO_KEEP:
                    curve_label = r'$k=%d$' % k
                    ysum = plotting_dict[plot_label][curve_label]['y_sum']
                    plotting_dict[plot_label][curve_label]['y_mean'] = ysum / num_runs

                    num_runs = plotting_dict[plot_label]['num_runs']
                    val_arr = np.array([plotting_dict[plot_label][curve_label]['runs'][r]
                                        for r in range(num_runs)])
                    #
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

        if separate_plots:
            if separate_by_init:
                for pc in PLOT_CODES_PRESENT:
                    plot_codes = [pc]
                    fmod = '_code_%s' % pc
                    plot_classify_dict(plotting_dict, K_TO_KEEP, compare_dir, plot_codes=plot_codes,
                                       xlim=xlim, legend=legend, fmod=fmod, figsize=figsize)
            else:
                for k in K_TO_KEEP:
                    k_to_show = [k]
                    fmod = '_k_%d' % k
                    plot_classify_dict(plotting_dict, k_to_show, compare_dir, plot_codes=PLOT_CODES_PRESENT,
                                       xlim=xlim, legend=legend, fmod=fmod, figsize=figsize)
        else:
            plot_classify_dict(plotting_dict, K_TO_KEEP, compare_dir, plot_codes=PLOT_CODES_PRESENT,
                               xlim=xlim, legend=legend, figsize=figsize)
