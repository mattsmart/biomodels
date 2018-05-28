import matplotlib.pyplot as plt
import numpy as np
import time
from os import sep
from multiprocessing import Pool, cpu_count

from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, DEFAULT_X_COLOUR, DEFAULT_Y_COLOUR, DEFAULT_Z_COLOUR
from data_io import read_varying_mean_sd_fpt_and_params, collect_fpt_mean_stats_and_params, read_fpt_and_params,\
                    write_fpt_and_params
from formulae import stoch_gillespie, get_physical_and_stable_fp

FS = 16
ec = 'k'  # 'k' or None
lw = 0.5
figsize = (8,6)

grey = (169 / 255.0, 169 / 255.0, 169 / 255.0)
blue = (119 / 255.0, 158 / 255.0, 203 / 255.0)
grey_dark = (149 / 255.0, 149 / 255.0, 149 / 255.0)
eps = 20.0 / 255.0
x_dark = [i - eps for i in DEFAULT_X_COLOUR]
z_dark = [i - eps for i in DEFAULT_Z_COLOUR]
# colours_orig = [DEFAULT_X_COLOUR, grey, DEFAULT_Z_COLOUR]
colours_dark = [x_dark, grey_dark, z_dark]
colours_blue = [x_dark, blue, z_dark]
colours = colours_blue

linspace_num_A = 80#70
linspace_num_B = 50 #75


def get_fpt(ensemble, init_cond, params, system, num_steps=100000):
    fp_times = np.zeros(ensemble)
    for i in xrange(ensemble):
        species, times = stoch_gillespie(init_cond, num_steps, system, params, fpt_flag=True)
        fp_times[i] = times[-1]
        print i, times[-1]
    return fp_times


def get_mean_fpt(init_cond, params, system, samplesize=32):
    fpt = get_fpt(samplesize, init_cond, params, system)
    return np.mean(fpt)


def wrapper_get_fpt(fn_args_dict):
    if fn_args_dict['kwargs'] is not None:
        return get_fpt(*fn_args_dict['args'], **fn_args_dict['kwargs'])
    else:
        return get_fpt(*fn_args_dict['args'])


def fast_fp_times(ensemble, init_cond, params, system, num_processes, num_steps='default'):
    if num_steps == 'default':
        kwargs_dict = {'num_steps': 100000}
    else:
        kwargs_dict = {'num_steps': num_steps}
    fn_args_dict = [0]*num_processes
    print "NUM_PROCESSES:", num_processes
    assert ensemble % num_processes == 0
    for i in xrange(num_processes):
        subensemble = ensemble / num_processes
        print "process:", i, "job size:", subensemble, "runs"
        fn_args_dict[i] = {'args': (subensemble, init_cond, params, system),
                           'kwargs': kwargs_dict}
    t0 = time.time()
    pool = Pool(num_processes)
    results = pool.map(wrapper_get_fpt, fn_args_dict)
    pool.close()
    pool.join()
    print "TIMER:", time.time() - t0

    fp_times = np.zeros(ensemble)
    for i, result in enumerate(results):
        fp_times[i*subensemble:(i+1)*subensemble] = result
    return fp_times


def map_init_name_to_init_cond(N, init_name):
    N = int(N)
    init_map = {"x_all": [N, 0, 0],
                "z_all": [0, 0, N],
                "midpoint": [N/3, N/3, N - 2*N/3],
                "z_close": [int(N*0.05), int(N*0.05), int(N*0.9)]}
    return init_map[init_name]


def fast_mean_fpt_varying(param_vary_name, param_vary_values, params, system, num_processes, init_name="x_all", samplesize=30):
    assert samplesize % num_processes == 0
    mean_fpt_varying = [0]*len(param_vary_values)
    sd_fpt_varying = [0]*len(param_vary_values)
    for idx, pv in enumerate(param_vary_values):
        params_step = params
        params_step[PARAMS_ID_INV[param_vary_name]] = pv
        N = params_step[PARAMS_ID_INV['N']]
        init_cond = map_init_name_to_init_cond(N, init_name)
        fp_times = fast_fp_times(samplesize, init_cond, params_step, system, num_processes)
        mean_fpt_varying[idx] = np.mean(fp_times)
        sd_fpt_varying[idx] = np.std(fp_times)
    return mean_fpt_varying, sd_fpt_varying


def fpt_histogram(fpt_list, params, system, show_flag=False, figname_mod="", x_log10_flag=False, y_log10_flag=False):
    ensemble_size = len(fpt_list)
    bins = np.linspace(np.min(fpt_list), np.max(fpt_list), 50)
    if x_log10_flag:
        max_log = np.ceil(np.max(np.log10(fpt_list)))
        plt.hist(fpt_list, bins=np.logspace(0.1, max_log, 50))
        ax = plt.gca()
        ax.set_xlabel('log10(fpt)')
        ax.set_ylabel('frequency')
        ax.set_xscale("log", nonposx='clip')
    elif y_log10_flag:
        plt.hist(fpt_list, bins=bins)
        ax = plt.gca()
        ax.set_xlabel('fpt')
        ax.set_ylabel('log10(frequency)')
        ax.set_yscale("log", nonposx='clip')
    else:
        plt.hist(fpt_list, bins=bins)
        ax = plt.gca()
        ax.set_xlabel('fpt')
        ax.set_ylabel('frequency')
    plt.title('First passage time histogram (%d runs) - %s' % (ensemble_size, system))
    # DRAW MEAN LINE
    plt.axvline(np.mean(fpt_list), color='k', linestyle='dashed', linewidth=2)
    # CREATE TABLE OF PARAMS
    row_labels = [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params[i]] for i in xrange(len(PARAMS_ID))]
    param_table = plt.table(cellText=table_vals,
                            colWidths=[0.1]*3,
                            rowLabels=row_labels,
                            loc='center right')
    plt_save = "fpt_histogram" + figname_mod
    plt.savefig(OUTPUT_DIR + sep + plt_save + '.png', bbox_inches='tight')
    if show_flag:
        plt.show()


def fpt_histogram_multi(multi_fpt_list, labels, show_flag=False, figname_mod="", x_log10_flag=False, y_log10_flag=False,
                        flag_norm=False, fs=12):

    # resize fpt lists if not all same size (to the min size)
    fpt_lengths = [len(fpt) for fpt in multi_fpt_list]
    ensemble_size = np.min(fpt_lengths)
    if sum(fpt_lengths - ensemble_size) > 0:
        print "Resizing multi_fpt_list elements:", fpt_lengths, "to the min size of:", ensemble_size
        for idx in xrange(len(fpt_lengths)):
            multi_fpt_list[idx] = multi_fpt_list[idx][:ensemble_size]
    bins = np.linspace(np.min(multi_fpt_list), np.max(multi_fpt_list), linspace_num_A)

    weights = None
    if flag_norm:
        y_label = 'Probability'
    else:
        y_label = 'Frequency'

    fig = plt.figure(figsize=figsize, dpi=120)
    if x_log10_flag:
        max_log = np.ceil(np.max(np.log10(multi_fpt_list)))
        for idx, fpt_list in enumerate(multi_fpt_list):
            if flag_norm:
                weights = np.ones_like(fpt_list) / ensemble_size
            plt.hist(fpt_list, bins=np.logspace(0.1, max_log, 100), alpha=1.0, color=colours[idx], label=labels[idx],weights=weights)
        ax = plt.gca()
        ax.set_xlabel('fpt', fontsize=fs)
        ax.set_ylabel(y_label, fontsize=fs)
        ax.set_xscale("log", nonposx='clip')
    elif y_log10_flag:
        for idx, fpt_list in enumerate(multi_fpt_list):
            if flag_norm:
                weights = np.ones_like(fpt_list) / ensemble_size
            plt.hist(fpt_list, bins=bins, alpha=0.6, color=colours[idx], label=labels[idx],weights=weights)#,edgecolor=ec,linewidth=lw)
            plt.hist(fpt_list, histtype='step', bins=bins, alpha=0.6, color=colours[idx], label=None,weights=weights, edgecolor=ec,
                     linewidth=lw, fill=False)

            #plt.hist(fpt_list, bins=bins, alpha=0.2, color=colours[idx], label=labels[idx],weights=weights)

        ax = plt.gca()
        ax.set_xlabel('First-passage time', fontsize=fs)
        ax.set_ylabel('%s' % y_label, fontsize=fs)
        ax.set_yscale("log", nonposx='clip')


        plt.locator_params(axis='x', nbins=4)

    else:
        for idx, fpt_list in enumerate(multi_fpt_list):
            if flag_norm:
                weights = np.ones_like(fpt_list) / ensemble_size
            #plt.hist(fpt_list, bins=bins, alpha=0.5, color=colours[idx], label=labels[idx],weights=weights,edgecolor=ec,linewidth=lw)

            plt.hist(fpt_list, bins=bins, alpha=0.6, color=colours[idx], label=labels[idx],weights=weights)#,edgecolor=ec,linewidth=lw)
            plt.hist(fpt_list, histtype='step', bins=bins, alpha=0.6, color=colours[idx], label=None,weights=weights, edgecolor=ec,
                     linewidth=lw, fill=False)

        ax = plt.gca()
        ax.set_xlabel('First-passage time (cell division timescale)', fontsize=fs)
        ax.set_ylabel(y_label, fontsize=fs)

    plt.title('First passage time histogram (%d runs)' % (ensemble_size))
    plt.legend(loc='upper right', fontsize=fs)
    ax.tick_params(labelsize=fs)
    # DRAW MEAN LINE
    #plt.axvline(np.mean(fpt_list), color='k', linestyle='dashed', linewidth=2)
    plt_save = "fpt_multihistogram" + figname_mod
    fig.savefig(OUTPUT_DIR + sep + plt_save + '.pdf', bbox_inches='tight')
    if show_flag:
        plt.show()


def fpt_histogram_multi_MAY25(multi_fpt_list, labels, show_flag=False, figname_mod="", x_log10_flag=False, y_log10_flag=False,
                        flag_norm=False, fs=12):

    # resize fpt lists if not all same size (to the min size)
    fpt_lengths = [len(fpt) for fpt in multi_fpt_list]
    ensemble_size = np.min(fpt_lengths)
    if sum(fpt_lengths - ensemble_size) > 0:
        print "Resizing multi_fpt_list elements:", fpt_lengths, "to the min size of:", ensemble_size
        for idx in xrange(len(fpt_lengths)):
            multi_fpt_list[idx] = multi_fpt_list[idx][:ensemble_size]
    bins = np.linspace(np.min(multi_fpt_list), np.max(multi_fpt_list), linspace_num_B)

    weights = None
    if flag_norm:
        y_label = 'probability'
    else:
        y_label = 'frequency'

    fig = plt.figure(figsize=figsize, dpi=120)

    if flag_norm:
        weights = np.ones_like(multi_fpt_list) / ensemble_size
    #plt.hist(multi_fpt_list, bins=bins, alpha=0.8, color=colours, label=labels, weights=weights, edgecolor=ec, linewidth=lw)
    plt.hist(multi_fpt_list, bins=bins, color=colours, label=labels, weights=weights, edgecolor=ec, linewidth=lw)

    ax = plt.gca()
    ax.set_xlabel('FP Time (cell division timescale)', fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    ax.set_yscale("log", nonposx='clip')
    plt.title('First passage time histogram (%d runs)' % (ensemble_size))
    plt.legend(loc='upper right', fontsize=fs)
    ax.tick_params(labelsize=fs)
    # DRAW MEAN LINE
    #plt.axvline(np.mean(fpt_list), color='k', linestyle='dashed', linewidth=2)
    plt_save = "fpt_multihistogram_may25" + figname_mod
    fig.savefig(OUTPUT_DIR + sep + plt_save + '.pdf', bbox_inches='tight')
    if show_flag:
        plt.show()


def plot_mean_fpt_varying(mean_fpt_varying, sd_fpt_varying, param_vary_name, param_set, params, system, samplesize, SEM_flag=True, show_flag=False, figname_mod=""):
    if SEM_flag:
        sd_fpt_varying = sd_fpt_varying / np.sqrt(samplesize)  # s.d. from CLT since sample mean is approx N(mu, sd**2/n)
    plt.errorbar(param_set, mean_fpt_varying, yerr=sd_fpt_varying, label="sim")
    plt.title("Mean FP Time, %s varying (sample=%d)" % (param_vary_name, samplesize))
    ax = plt.gca()
    ax.set_xlabel(param_vary_name)
    ax.set_ylabel('Mean FP time')
    # CREATE TABLE OF PARAMS
    row_labels = [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params[i]] if PARAMS_ID[i] not in [param_vary_name] else ["None"]
                  for i in xrange(len(PARAMS_ID))]
    param_table = plt.table(cellText=table_vals,
                            colWidths=[0.1]*3,
                            rowLabels=row_labels,
                            loc='center right')
    plt_save = "mean_fpt_varying" + figname_mod
    plt.savefig(OUTPUT_DIR + sep + plt_save + '.png', bbox_inches='tight')
    if show_flag:
        plt.show()
    return ax


if __name__ == "__main__":
    # SCRIPT FLAGS
    flag_compute_fpt = False
    flag_read_fpt = False
    flag_generate_hist_multi = False
    flag_load_hist_multi = True
    flag_collect = False
    flag_means_read_and_plot = False
    flag_means_collect_and_plot = False
    flag_seaborn = False

    # SCRIPT PARAMETERS
    system = "feedback_z"  # "feedback_mu_XZ_model" or "feedback_z"
    num_steps = 100000  # default 100000
    ensemble = 5  # default 100

    # DYNAMICS PARAMETERS
    alpha_plus = 0.2 #0.2  # 0.05 #0.4
    alpha_minus = 0.5 #0.5  # 4.95 #0.5
    mu = 0.001  # 0.001
    a = 1.0
    b = 0.8
    c = 0.81  # 2.6 #1.2
    N = 100.0  # 100
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    mu_base = mu*1e-1
    params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]

    # OTHER PARAMETERS
    init_cond = [int(N), 0, 0]

    if flag_compute_fpt:
        fp_times = get_fpt(ensemble, init_cond, num_steps, params, system)
        write_fpt_and_params(fp_times, params, system)
        fpt_histogram(fp_times, params, system, show_flag=True, figname_mod="XZ_model_withFeedback_mu1e-1")

    if flag_read_fpt:
        dbdir = OUTPUT_DIR
        dbdir_100 = dbdir + sep + "fpt_mean" + sep + "100_c95"
        fp_times_xyz_100, params_a, system_a = read_fpt_and_params(dbdir_100)
        dbdir_10k = dbdir + sep + "fpt_mean" + sep + "10k_c95"
        fp_times_xyz_10k, params_b, system_b = read_fpt_and_params(dbdir_10k)

    if flag_generate_hist_multi:
        ensemble = 14
        num_proc = cpu_count() - 1
        param_vary_id = "c"
        param_idx = PARAMS_ID_INV[param_vary_id]
        param_vary_values = [0.8, 0.88, 0.95]
        param_vary_labels = ['Region I','Region IV','Region III']
        params_set = [params[:] for _ in param_vary_values]
        multi_fpt = np.zeros((len(param_vary_values), ensemble))
        multi_fpt_labels = ['label' for _ in param_vary_values]
        for idx, param_val in enumerate(param_vary_values):
            param_val_string = "%s=%.3f" % (param_vary_id, param_val)
            params_set[idx][param_idx] = param_val
            #fp_times = get_fpt(ensemble, init_cond, params_set[idx], system, num_steps=num_steps)
            fp_times = fast_fp_times(ensemble, init_cond, params_set[idx], system, num_proc, num_steps=num_steps)
            write_fpt_and_params(fp_times, params_set[idx], system, filename="fpt_multi", filename_mod=param_val_string)
            multi_fpt[idx,:] = np.array(fp_times)
            multi_fpt_labels[idx] = "%s (%s)" % (param_vary_labels[idx], param_val_string)
        fpt_histogram_multi(multi_fpt, multi_fpt_labels, show_flag=True, y_log10_flag=False)

    if flag_load_hist_multi:
        flag_norm = True
        dbdir = OUTPUT_DIR + sep + "may25_100"
        #dbdir_c80 = dbdir + "fpt_feedback_z_ens1040_c0.80_params"
        c80_header = "fpt_feedback_z_ens1040_c80_N100"
        c88_header = "fpt_feedback_z_ens1040_c88_N100"
        c95_header = "fpt_feedback_z_ens1040_c95_N100"
        fp_times_xyz_c80, params_a, system_a = read_fpt_and_params(dbdir, "%s_data.txt" % c80_header, "%s_params.csv" % c80_header)
        fp_times_xyz_c88, params_b, system_b = read_fpt_and_params(dbdir, "%s_data.txt" % c88_header, "%s_params.csv" % c88_header)
        fp_times_xyz_c95, params_c, system_c = read_fpt_and_params(dbdir, "%s_data.txt" % c95_header, "%s_params.csv" % c95_header)
        #fpt_histogram(fp_times_xyz_c88, params_b, system_b, y_log10_flag=False, figname_mod="_xyz_feedbackz_N10k_c88_may25")
        #plt.close('all')
        #fpt_histogram(fp_times_xyz_c88, params_b, system_b, y_log10_flag=True, figname_mod="_xyz_feedbackz_N10k_c88_may25_logy")
        #plt.close('all')
        multi_fpt = [fp_times_xyz_c80, fp_times_xyz_c88, fp_times_xyz_c95]
        labels = ("c=0.80 (Region I)", "c=0.88 (Region IV)", "c=0.95 (Region III)")
        fpt_histogram_multi(multi_fpt, labels, show_flag=True, y_log10_flag=False, flag_norm=flag_norm, fs=FS)
        #plt.close('all')
        fpt_histogram_multi(multi_fpt, labels, show_flag=True, y_log10_flag=True, flag_norm=flag_norm, fs=FS)
        #fpt_histogram_multi_MAY25(multi_fpt, labels, show_flag=True, y_log10_flag=True, flag_norm=flag_norm, fs=FS)

    if flag_seaborn:

        dbdir = OUTPUT_DIR + sep + "may25_100"
        #dbdir_c80 = dbdir + "fpt_feedback_z_ens1040_c0.80_params"
        c80_header = "fpt_feedback_z_ens1040_c80_N100"
        c88_header = "fpt_feedback_z_ens1040_c88_N100"
        c95_header = "fpt_feedback_z_ens1040_c95_N100"
        fp_times_xyz_c80, params_a, system_a = read_fpt_and_params(dbdir, "%s_data.txt" % c80_header, "%s_params.csv" % c80_header)
        fp_times_xyz_c88, params_b, system_b = read_fpt_and_params(dbdir, "%s_data.txt" % c88_header, "%s_params.csv" % c88_header)
        fp_times_xyz_c95, params_c, system_c = read_fpt_and_params(dbdir, "%s_data.txt" % c95_header, "%s_params.csv" % c95_header)
        #fpt_histogram(fp_times_xyz_c88, params_b, system_b, y_log10_flag=False, figname_mod="_xyz_feedbackz_N10k_c88_may25")
        #plt.close('all')
        #fpt_histogram(fp_times_xyz_c88, params_b, system_b, y_log10_flag=True, figname_mod="_xyz_feedbackz_N10k_c88_may25_logy")
        #plt.close('all')
        multi_fpt = [fp_times_xyz_c80, fp_times_xyz_c88, fp_times_xyz_c95]
        labels = ("XYZ_c0.80_N100", "XYZ_c0.88_N100", "XYZ_c0.95_N100")

        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # Create the data
        fpt_array = np.array(multi_fpt)

        df = pd.DataFrame(dict(g = labels, dist = multi_fpt))

        # Create the data
        #x = rs.randn(90)
        ensemble=1040
        x = np.reshape(fpt_array, 3120, order='F')
        tiles = np.tile(list("ABC"), ensemble)
        df = pd.DataFrame(dict(x=x, g=tiles))
        m = df.g.map(ord)
        df["x"] += m

        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(df, row="g", hue="g", aspect=15, size=.5, palette=pal)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
        g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
        g.map(plt.axhline, y=0, lw=2, clip_on=False)


        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)


        g.map(label, "x")

        # Set the subplots to overlap
        g.fig.subplots_adjust(hspace=-.25)

        # Remove axes details that don't play will with overlap
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=True, left=True)

        plt.show()




    if flag_means_read_and_plot:
        datafile = OUTPUT_DIR + sep + "fpt_stats_collected_mean_sd_varying_N.txt"
        paramfile = OUTPUT_DIR + sep + "fpt_stats_collected_mean_sd_varying_N_params.csv"
        samplesize=48
        mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, system = \
            read_varying_mean_sd_fpt_and_params(datafile, paramfile)
        plt_axis = plot_mean_fpt_varying(mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, system, samplesize,
                                         SEM_flag=True, show_flag=True, figname_mod="_%s_n%d" % (param_to_vary, samplesize))
        """
        mu = params[PARAMS_ID_INV['mu']]
        mixed_fp_zinf_at_N = [0.0]*len(param_set)
        for idx, N in enumerate(param_set):
            params_at_N = params
            params_at_N[PARAMS_ID_INV['N']] = N
            fps = get_physical_and_stable_fp(params_at_N, system)
            assert len(fps) == 1
            mixed_fp_zinf_at_N[idx] = fps[0][2]
        plt_axis.plot(param_set, [1/(mu*n) for n in param_set], '-o', label="(mu*N)^-1")
        plt_axis.plot(param_set, [1/(mu*zinf) for zinf in mixed_fp_zinf_at_N], '-o', label="(mu*z_inf)^-1")
        plt_axis.set_yscale("log", nonposx='clip')
        plt_axis.set_xscale("log", nonposx='clip')
        plt_axis.legend()
        plt.savefig(OUTPUT_DIR + sep + "theorycompare_loglog" + '.png', bbox_inches='tight')
        plt.show()
        """

    if flag_means_collect_and_plot:
        dbdir = OUTPUT_DIR + sep + "tocollect" + sep + "runset_nov4_2pm_c6_Nzclose"
        datafile, paramfile = collect_fpt_mean_stats_and_params(dbdir)
        samplesize=48
        mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, system = \
            read_varying_mean_sd_fpt_and_params(datafile, paramfile)
        plot_mean_fpt_varying(mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, system, samplesize,
                              SEM_flag=True, show_flag=True, figname_mod="_%s_n%d" % (param_to_vary, samplesize))
