import csv
import numpy as np
from os import sep, listdir, mkdir, makedirs
from os.path import join, isfile, isdir, basename, dirname, exists

from constants import PARAMS_ID, OUTPUT_DIR, CSV_DATA_TYPES, PARAMS_ID_INV, ODE_SYSTEMS
from params import Params


def write_bifurc_data(bifurcation_search, x0, x0_stab, x1, x1_stab, x2, x2_stab, bifurc_id, filedir, filename):
    csv_header = [bifurc_id, 'x0_x', 'x0_y', 'x0_z', 'x0_stab', 'x1_x', 'x1_y', 'x1_z', 'x1_stab', 'x2_x', 'x2_y',
                  'x2_z', 'x2_stab']
    filepath = filedir + sep + filename
    with open(filepath, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(csv_header)
        for idx in xrange(len(bifurcation_search)):
            line = [bifurcation_search[idx]] + list(x0[idx,:]) + list(x0_stab[idx]) + list(x1[idx,:]) + \
                   list(x1_stab[idx]) + list(x2[idx,:]) + list(x2_stab[idx])
            writer.writerow(line)
    return filepath


def write_params(params, filedir, filename):
    return params.write(filedir, filename)


def read_params(filedir, filename):
    return Params.read(filedir, filename)


def read_bifurc_data(filedir, filename):
    def str_to_data(elem):
        if elem == 'True':
            return True
        elif elem == 'False':
            return False
        else:
            return elem
    with open(filedir + sep + filename, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        nn = sum(1 for row in datareader) - 1
        csvfile.seek(0)
        header = datareader.next()
        data_dict = {key: np.zeros((nn, 1), dtype=CSV_DATA_TYPES[key]) for key in header}
        for idx_row, row in enumerate(datareader):
            for idx_col, elem in enumerate(row):
                data_dict[header[idx_col]][idx_row] = str_to_data(elem)
    return data_dict


def write_fpt_and_params(fp_times, fp_states, params, filedir=OUTPUT_DIR, filename="fpt", filename_mod=""):
    if filename_mod != "":
        filename_params = filename + "_" + filename_mod + "_params.csv"
        filename_fp_times = filename + "_" + filename_mod + "_times.txt"
        filename_fp_states = filename + "_" + filename_mod + "_states.txt"
    else:
        filename_params = filename + "_params.csv"
        filename_fp_times = filename + "_times.txt"
        filename_fp_states = filename + "_states.txt"
    params.write(filedir, filename_params)
    with open(filedir + sep + filename_fp_times, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for idx in xrange(len(fp_times)):
            writer.writerow([str(fp_times[idx])])
    if fp_states is not None:
        with open(filedir + sep + filename_fp_states, "wb") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for idx in xrange(len(fp_states)):
                writer.writerow(fp_states[idx, :])
    return filedir + sep + filename_fp_times


def write_varying_mean_sd_fpt_and_params(fpt_mean, fpt_sd, param_vary_name, param_vary_set, params, filedir=OUTPUT_DIR, filename="fpt_stats", filename_mod=""):
    if filename_mod != "":
        filename_params = filename + "_" + filename_mod + "_mean_sd_varying_%s_params.csv" % param_vary_name
        filename_fpt = filename + "_" + filename_mod + "_mean_sd_varying_%s.txt" % param_vary_name
    else:
        filename_params = filename + "_mean_sd_varying_%s_params.csv" % param_vary_name
        filename_fpt = filename + "_mean_sd_varying_%s.txt" % param_vary_name
    params = params.mod_copy({param_vary_name: None})
    params.write(filedir, filename_params)
    with open(filedir + sep + filename_fpt, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([param_vary_name, "fpt_mean", "fpt_sd"])
        for idx in xrange(len(fpt_mean)):
            datarow = [str(param_vary_set[idx]), str(fpt_mean[idx]), str(fpt_sd[idx])]
            writer.writerow(datarow)
    return filedir + sep + filename_fpt, filedir + sep + filename_params


def read_fpt_and_params(filedir, filename_times=None, filename_states=None, filename_params=None):
    onlyfiles = [f for f in listdir(filedir) if isfile(join(filedir, f))]
    timefiles = [f for f in onlyfiles if "times" == f[-9:-4]]
    statefiles = [f for f in onlyfiles if "states" == f[-10:-4]]
    paramfiles = [f for f in onlyfiles if "params" == f[-10:-4]]
    if filename_times is None:
        assert len(timefiles) == 1
        filename_times = basename(timefiles[0])
    if filename_states is None:
        if len(statefiles) == 1:
            filename_states = basename(statefiles[0])
    if filename_params is None:
        assert len(paramfiles) == 1
        filename_params = basename(paramfiles[0])

    params = read_params(filedir, filename_params)
    assert params.system in ODE_SYSTEMS
    with open(filedir + sep + filename_times, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        nn = sum(1 for row in datareader)
        csvfile.seek(0)
        fp_times = [0.0]*nn
        for idx, fpt in enumerate(datareader):
            fp_times[idx] = float(fpt[0])
    if filename_states is None:
        fp_states = None
    else:
        with open(filedir + sep + filename_states, 'rb') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
            nn = sum(1 for row in datareader)
            csvfile.seek(0)
            fp_states = np.zeros((nn, params.numstates))
            for idx, row in enumerate(datareader):
                fp_states[idx,:] = [float(a) for a in row]
    return fp_times, fp_states, params


def read_varying_mean_sd_fpt(datafile):
    with open(datafile, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        param_vary_name = next(datareader)[0]
        nn = sum(1 for row in datareader)
        csvfile.seek(0)
        next(datareader)  # skip header line
        param_vary_list = [0.0] * nn
        fpt_means = [0.0] * nn
        fpt_sd = [0.0] * nn
        for idx, fpt in enumerate(datareader):
            param_vary_list[idx] = float(fpt[0])
            fpt_means[idx] = float(fpt[1])
            fpt_sd[idx] = float(fpt[2])
    return fpt_means, fpt_sd, param_vary_name, param_vary_list


def read_varying_mean_sd_fpt_and_params(datafile, paramfile):
    params = Params.read(dirname(paramfile), basename(paramfile))
    mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set = read_varying_mean_sd_fpt(datafile)
    return mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params


def collect_fpt_and_params_deprecated(filedir):
    onlydirs = [join(filedir, subdir) for subdir in listdir(filedir) if isdir(join(filedir, subdir))]
    timesfiles = []
    paramfiles = []
    for subdir in onlydirs:
        files = listdir(subdir)
        for f in files:
            if "data" == f[-8:-4]:
                timesfiles.append(join(subdir, f))
            if "params" == f[-10:-4]:
                paramfiles.append(join(subdir, f))
    assert len(timesfiles) == len(paramfiles)

    params_0 = read_params(dirname(paramfiles[0]), basename(paramfiles[0]))
    for pf in paramfiles:
        params = read_params(dirname(pf), basename(pf))
        assert params.params_list == params_0.params_list

    fpt_collected = []
    for idx, df in enumerate(timesfiles):
        fp_times, fp_states, params = read_fpt_and_params(dirname(df), filename_times=basename(df), filename_params=basename(paramfiles[0]))
        fpt_collected += fp_times

    collected_dirname = "collected_%d" % len(fpt_collected)
    collected_dir = filedir + sep + collected_dirname
    mkdir(collected_dir)
    print 'TODO implement fp_states from None in collect_fpt_and_params()'
    write_fpt_and_params(fpt_collected, None, params_0, filedir=collected_dir, filename="fpt", filename_mod=collected_dirname)
    return collected_dir


def collect_fpt_and_params(filedir):
    #TODO collect states also
    onlydirs = [join(filedir, subdir) for subdir in listdir(filedir) if isdir(join(filedir, subdir))]
    timesfiles = []
    statesfiles = []
    paramfiles = []
    for subdir in onlydirs:
        files = listdir(subdir)
        for f in files:
            if "times" == f[-9:-4]:
                timesfiles.append(join(subdir, f))
            if "states" == f[-10:-4]:
                statesfiles.append(join(subdir, f))
            if "params" == f[-10:-4]:
                paramfiles.append(join(subdir, f))
    assert len(timesfiles) == len(paramfiles)
    assert len(timesfiles) == len(statesfiles)

    params_0 = read_params(dirname(paramfiles[0]), basename(paramfiles[0]))
    for pf in paramfiles:
        params = read_params(dirname(pf), basename(pf))
        assert params.params_list == params_0.params_list

    fpt_collected = []
    fpstate_collected = np.empty((0, params_0.numstates))
    for idx, df in enumerate(timesfiles):
        fp_times, fp_states, params = read_fpt_and_params(dirname(df), filename_times=basename(df),
                                                          filename_params=basename(paramfiles[0]))
        fpt_collected += fp_times
        fpstate_collected = np.append(fpstate_collected, fp_states, axis=0)  # TODO check form is correct
    assert fpstate_collected.shape[0] == len(fpt_collected)
    assert fpstate_collected.shape[1] == 3

    collected_dirname = "collected_%d" % len(fpt_collected)
    collected_dir = filedir + sep + collected_dirname
    mkdir(collected_dir)
    print 'TODO implement fp_states from None in collect_fpt_and_params()'
    write_fpt_and_params(fpt_collected, fpstate_collected, params_0, filedir=collected_dir, filename="fpt",
                         filename_mod=collected_dirname)
    return collected_dir


def collect_fpt_mean_stats_and_params(filedir, dirbase="means"):
    # NOTE: assumes folder structure is filedir -> means1, means2, means3 ... collection of dirs
    #       each of these dirs contains an output folder with a params file and a data file

    def data_and_param_files_from_fptdir(fptdir):
        # input: location of filedir/means1 for example
        outputdir = fptdir + sep + "output"
        outputdirfiles = listdir(outputdir)
        #assert len(outputdirfiles) == 2
        for f in outputdirfiles:
            if f[0:9] == "fpt_stats":
                if f[-10:-4] == "params":
                    paramfile = f
                else:
                    datafile = f
        return outputdir + sep + datafile, outputdir + sep + paramfile

    onlydirs = [f for f in listdir(filedir) if isdir(join(filedir, f))]
    dirstocheck = [filedir + sep + fptdir for fptdir in onlydirs if basename(fptdir)[:len(dirbase)] == dirbase]
    dirstocheck.sort(key=lambda f: int(filter(str.isdigit, f)))  # assumes dirbase+suffix (e.g.'means7') has only 1 int

    _, pf = data_and_param_files_from_fptdir(dirstocheck[0])
    print pf
    print dirname(pf), basename(pf)
    params_0 = read_params(dirname(pf), basename(pf))

    fpt_means_collected = []
    fpt_sd_collected = []
    param_vary_collected = []
    for idx, fptdir in enumerate(dirstocheck):
        df, pf = data_and_param_files_from_fptdir(fptdir)
        params = read_params(dirname(pf), basename(pf))
        assert params.params_list == params_0.params_list  # TODO alt to this is implement __eq__ method in obj
        assert params.system == params_0.system
        assert params.feedback == params_0.feedback
        fpt_means, fpt_sd, param_vary_name, param_vary_list = read_varying_mean_sd_fpt(df)
        fpt_means_collected += fpt_means
        fpt_sd_collected += fpt_sd
        param_vary_collected += param_vary_list

    coll_dirname = "collected_stats_%d" % len(fpt_means_collected)
    collected_dir = filedir + sep + coll_dirname
    if not exists(collected_dir):
        makedirs(collected_dir)
    coll_df, coll_pf = write_varying_mean_sd_fpt_and_params(fpt_means_collected, fpt_sd_collected, param_vary_name, param_vary_collected,
                                                            params_0, filedir=collected_dir, filename_mod="collected")
    return coll_df, coll_pf


def write_matrix_data_and_idx_vals(arr, row_vals, col_vals, dataname, rowname, colname, output_dir=OUTPUT_DIR, binary=False):
    datapath = output_dir + sep + dataname
    rowpath = output_dir + sep + dataname + '_' + rowname
    colpath = output_dir + sep + dataname + '_' + colname
    if binary:
        np.save(datapath, np.array(arr))
        np.save(rowpath, np.array(row_vals))
        np.save(colpath, np.array(col_vals))
    else:
        np.savetxt(datapath + '.txt', np.array(arr), delimiter=",")
        np.savetxt(rowpath + '.txt', np.array(row_vals), delimiter=",")
        np.savetxt(colpath + '.txt', np.array(col_vals), delimiter=",")
    return datapath, rowpath, colpath


def read_matrix_data_and_idx_vals(datapath, rowpath, colpath, binary=False):
    if binary:
        arr = np.load(datapath)
        row = np.load(rowpath)
        col = np.load(colpath)
    else:
        arr = np.loadtxt(datapath, delimiter=",", dtype=float)
        row = np.loadtxt(rowpath, delimiter=",", dtype=float)
        col = np.loadtxt(colpath, delimiter=",", dtype=float)
    return arr, row, col


if __name__ == '__main__':
    collect_fpt = True
    collect_means = False
    collect_multiple_means = False

    if collect_fpt:
        collect_dir = OUTPUT_DIR + sep + 'tocollect' + sep + 'varygamma_MP2_june13'
        collect_fpt_and_params(collect_dir)

    if collect_means:
        subdirs = []
        collect_dirs = [OUTPUT_DIR + sep + 'tocollect' + sep + 'varygamma_MP2_june13_p2' + sep + a for a in subdirs]
        for collect_dir in collect_dirs:
            collect_fpt_mean_stats_and_params(collect_dir, dirbase="means")

    if collect_multiple_means:
        dirname = OUTPUT_DIR + sep + 'tocollect' + sep + 'varygamma_MP2_june13_p2'
        subdirs = [join(dirname, f) for f in listdir(dirname) if isdir(join(dirname, f))]
        means_data = {}  # form is nval key to {ens, mean}
        ENS = 72
        for subdir in subdirs:
            print subdir
            df = subdir + sep + 'fpt_stats_collected_mean_sd_varying_gamma.txt'
            mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set = read_varying_mean_sd_fpt(df)
            for idx, elem in enumerate(mean_fpt_varying):
                Nval = param_set[idx]
                meanval = mean_fpt_varying[idx]
                sdval = sd_fpt_varying[idx]

                print 'dataline', Nval, meanval, sdval

                if Nval in means_data.keys():
                    print Nval
                    print means_data[Nval]
                    means_data[Nval][1] = (means_data[Nval][0] * means_data[Nval][1] + meanval) / (1 + means_data[Nval][0])
                    means_data[Nval][2] = np.sqrt((means_data[Nval][0] * (means_data[Nval][1] ** 2) + sdval ** 2) / \
                                          (1 + means_data[Nval][0]))
                    means_data[Nval][0] += 1
                else:
                    means_data[Nval] = np.array([1, meanval, sdval])

        nsorted = sorted(means_data.keys())
        for Nval in nsorted:
            print Nval, means_data[Nval][0], means_data[Nval][1], means_data[Nval][2]

        print 'formatted'
        for Nval in nsorted:
            print '%.2f,%.8f,%.8f' % (Nval, means_data[Nval][1], means_data[Nval][2])
