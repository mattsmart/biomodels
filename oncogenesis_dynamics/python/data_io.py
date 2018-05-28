import csv
import numpy as np
from os import sep, listdir, mkdir
from os.path import join, isfile, isdir, basename, dirname

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


def write_fpt_and_params(fpt, params, filedir=OUTPUT_DIR, filename="fpt", filename_mod=""):
    if filename_mod != "":
        filename_params = filename + "_" + filename_mod + "_params.csv"
        filename_fpt = filename + "_" + filename_mod + "_data.txt"
    else:
        filename_params = filename + "_params.csv"
        filename_fpt = filename + "_data.txt"
    params.write(filedir, filename_params)
    with open(filedir + sep + filename_fpt, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for idx in xrange(len(fpt)):
            writer.writerow([str(fpt[idx])])
    return filedir + sep + filename_fpt


def write_varying_mean_sd_fpt_and_params(fpt_mean, fpt_sd, param_vary_name, param_vary_set, params, filedir=OUTPUT_DIR, filename="fpt_stats", filename_mod=""):
    if filename_mod != "":
        filename_params = filename + "_" + filename_mod + "_mean_sd_varying_%s_params.csv" % param_vary_name
        filename_fpt = filename + "_" + filename_mod + "_mean_sd_varying_%s.txt" % param_vary_name
    else:
        filename_params = filename + "_mean_sd_varying_%s_params.csv" % param_vary_name
        filename_fpt = filename + "_mean_sd_varying_%s.txt" % param_vary_name
    params = params.mod_copy([(param_vary_name, None)])
    params.write(filedir, filename_params)
    with open(filedir + sep + filename_fpt, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([param_vary_name, "fpt_mean", "fpt_sd"])
        for idx in xrange(len(fpt_mean)):
            datarow = [str(param_vary_set[idx]), str(fpt_mean[idx]), str(fpt_sd[idx])]
            writer.writerow(datarow)
    return filedir + sep + filename_fpt, filedir + sep + filename_params


def read_fpt_and_params(filedir, filename_data=None, filename_params=None):
    onlyfiles = [f for f in listdir(filedir) if isfile(join(filedir, f))]
    datafiles = [f for f in onlyfiles if "data" == f[-8:-4]]
    paramfiles = [f for f in onlyfiles if "params" == f[-10:-4]]
    if filename_data is None:
        assert len(datafiles) == 1
        filename_data = basename(datafiles[0])
    if filename_params is None:
        assert len(paramfiles) == 1
        filename_params = basename(paramfiles[0])

    params = read_params(filedir, filename_params)
    assert params.system in ODE_SYSTEMS
    with open(filedir + sep + filename_data, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        nn = sum(1 for row in datareader)
        csvfile.seek(0)
        fp_times = [0.0]*nn
        for idx, fpt in enumerate(datareader):
            fp_times[idx] = float(fpt[0])
    return fp_times, params


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


def collect_fpt_and_params(filedir):
    # NOTE: assumes folder structure 8s N files of ..._data and N files of ..._params which ALL correspond
    onlydirs = [f for f in listdir(filedir) if isfile(join(filedir, f))]
    datafiles = [f for f in onlydirs if "data" == f[-8:-4]]
    paramfiles = [f for f in onlydirs if "params" == f[-10:-4]]
    assert len(datafiles) == len(paramfiles)

    params_0 = read_params(filedir, basename(paramfiles[0]))
    for pf in paramfiles:
        params = read_params(filedir, basename(pf))
        assert params == params_0

    fpt_collected = []
    for idx, df in enumerate(datafiles):
        fp_times, params = read_fpt_and_params(filedir, df, basename(paramfiles[0]))
        fpt_collected += fp_times

    collected_dirname = "collected_%d" % len(fpt_collected)
    collected_dir = filedir + sep + collected_dirname
    mkdir(collected_dir)
    write_fpt_and_params(fpt_collected, params_0, filedir=collected_dir, filename="fpt", filename_mod=collected_dirname)
    return collected_dir


def collect_fpt_mean_stats_and_params(filedir, dirbase="means"):
    # NOTE: assumes folder structure is filedir -> means1, means2, means3 ... collection of dirs
    #       each of these dirs contains an output folder with a params file and a data file

    def data_and_param_files_from_fptdir(fptdir):
        # input: location of filedir/means1 for example
        outputdir = fptdir + sep + "output"
        outputdirfiles = listdir(outputdir)
        assert len(outputdirfiles) == 2
        for f in outputdirfiles:
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
        assert params == params_0
        fpt_means, fpt_sd, param_vary_name, param_vary_list = read_varying_mean_sd_fpt(df)
        fpt_means_collected += fpt_means
        fpt_sd_collected += fpt_sd
        param_vary_collected += param_vary_list

    coll_dirname = "collected_stats_%d" % len(fpt_means_collected)
    collected_dir = filedir + sep + coll_dirname
    mkdir(collected_dir)
    coll_df, coll_pf = write_varying_mean_sd_fpt_and_params(fpt_means_collected, fpt_sd_collected, param_vary_name, param_vary_collected,
                                                            params_0, filedir=collected_dir, filename_mod="collected")
    return coll_df, coll_pf


def write_matrix_data_and_idx_vals(arr, row_vals, col_vals, dataname, rowname, colname, output_dir=OUTPUT_DIR):
    datapath = output_dir + sep + dataname + ".txt"
    rowpath = output_dir + sep + dataname + '_' + rowname + ".txt"
    colpath = output_dir + sep + dataname + '_' + colname + ".txt"
    np.savetxt(datapath, np.array(arr), delimiter=",")
    np.savetxt(rowpath, np.array(row_vals), delimiter=",")
    np.savetxt(colpath, np.array(col_vals), delimiter=",")
    return datapath, rowpath, colpath


def read_matrix_data_and_idx_vals(datapath, rowpath, colpath):
    arr = np.loadtxt(datapath, delimiter=",", dtype=float)
    row = np.loadtxt(rowpath, delimiter=",", dtype=float)
    col = np.loadtxt(colpath, delimiter=",", dtype=float)
    return arr, row, col
