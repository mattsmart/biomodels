import csv
from os import sep, listdir, mkdir
from os.path import join, isfile, basename, dirname

from constants import PARAMS_ID, OUTPUT_DIR, CSV_DATA_TYPES, PARAMS_ID_INV, ODE_SYSTEMS


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


def write_params(params, system, filedir, filename):
    filepath = filedir + sep + filename
    with open(filepath, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for idx in xrange(len(PARAMS_ID)):
            if params[idx] is None:
                params[idx] = 'None'
            writer.writerow([PARAMS_ID[idx], params[idx]])
        # any extra non-dynamics params
        writer.writerow(['system', system])
    return filepath


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


def read_params(filedir, filename):
    with open(filedir + sep + filename, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        num_params = sum(1 for row in datareader)
        csvfile.seek(0)
        params = [0.0]*num_params
        for idx, pair in enumerate(datareader):
            if idx < num_params - 1:
                assert pair[0] == PARAMS_ID[idx]
                if pair[1] != 'None':
                    params[idx] = float(pair[1])
                else:
                    params[idx] = None
            else:
                assert pair[0] == 'system'
                params[idx] = pair[1]
    return params


def write_fpt_and_params(fpt, params, system, filedir=OUTPUT_DIR, filename="fpt", filename_mod=""):
    if filename_mod != "":
        filename_params = filename + "_" + filename_mod + "_params.csv"
        filename_fpt = filename + "_" + filename_mod + "_data.txt"
    else:
        filename_params = filename + "_params.csv"
        filename_fpt = filename + "_data.txt"
    write_params(params, system, filedir, filename_params)
    with open(filedir + sep + filename_fpt, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for idx in xrange(len(fpt)):
            writer.writerow([str(fpt[idx])])
    return filedir + sep + filename_fpt


def write_varying_mean_sd_fpt_and_params(fpt_mean, fpt_sd, param_vary_name, param_vary_set, params, system, filedir=OUTPUT_DIR, filename="fpt_stats", filename_mod=""):
    if filename_mod != "":
        filename_params = filename + "_" + filename_mod + "_mean_sd_varying_%s_params.csv" % param_vary_name
        filename_fpt = filename + "_" + filename_mod + "_mean_sd_varying_%s.txt" % param_vary_name
    else:
        filename_params = filename + "_mean_sd_varying_%s_params.csv" % param_vary_name
        filename_fpt = filename + "_mean_sd_varying_%s.txt" % param_vary_name
    params[PARAMS_ID_INV[param_vary_name]] = None
    write_params(params, system, filedir, filename_params)
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

    params_with_system = read_params(filedir, filename_params)
    assert params_with_system[-1] in ODE_SYSTEMS
    params = params_with_system[:-1]
    system = params_with_system[-1]
    with open(filedir + sep + filename_data, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        nn = sum(1 for row in datareader)
        csvfile.seek(0)
        fp_times = [0.0]*nn
        for idx, fpt in enumerate(datareader):
            fp_times[idx] = float(fpt[0])
    return fp_times, params, system


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
    params = read_params(dirname(paramfile), basename(paramfile))
    mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set = read_varying_mean_sd_fpt(datafile)
    return mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params[:-1], params[-1]


def collect_fpt_and_params(filedir):
    # NOTE: assumes folder structure 8s N files of ..._data and N files of ..._params which ALL correspond
    onlyfiles = [f for f in listdir(filedir) if isfile(join(filedir, f))]
    datafiles = [f for f in onlyfiles if "data" == f[-8:-4]]
    paramfiles = [f for f in onlyfiles if "params" == f[-10:-4]]
    assert len(datafiles) == len(paramfiles)

    params_0 = read_params(filedir, basename(paramfiles[0]))
    for pf in paramfiles:
        params = read_params(filedir, basename(pf))
        assert params == params_0

    fpt_collected = []
    for idx, df in enumerate(datafiles):
        fp_times, params, system = read_fpt_and_params(filedir, df, basename(paramfiles[0]))
        fpt_collected += fp_times

    dirname = "collected_%d" % len(fpt_collected)
    collected_dir = filedir + sep + dirname
    mkdir(collected_dir)
    write_fpt_and_params(fpt_collected, params_0[:-1], params_0[-1], filedir=collected_dir, filename="fpt", filename_mod=dirname)
    return collected_dir


def collect_fpt_mean_stats_and_params(filedir):
    # NOTE: assumes folder structure 8s N files of ..._data and N files of ..._params which ALL correspond
    onlyfiles = [f for f in listdir(filedir) if isfile(join(filedir, f))]
    datafiles = [f for f in onlyfiles if "data" == f[-8:-4]]
    paramfiles = [f for f in onlyfiles if "params" == f[-10:-4]]
    assert len(datafiles) == len(paramfiles)

    params_0 = read_params(filedir, basename(paramfiles[0]))
    for pf in paramfiles:
        params = read_params(filedir, basename(pf))
        assert params == params_0

    fpt_means_collected = []
    fpt_sd_collected = []
    param_vary_collected = []
    for idx, df in enumerate(datafiles):
        fpt_means, fpt_sd, param_vary_name, param_vary_list = read_varying_mean_sd_fpt(filedir + sep + df)
        fpt_means_collected += fpt_means
        fpt_sd_collected += fpt_sd
        param_vary_collected += param_vary_list

    dirname = "collected_stats_%d" % len(fpt_means_collected)
    collected_dir = filedir + sep + dirname
    mkdir(collected_dir)
    write_varying_mean_sd_fpt_and_params(fpt_means_collected, fpt_sd_collected, param_vary_name, param_vary_collected,
                                         params_0[:-1], params_0[-1], filedir=collected_dir, filename_mod="collected")
    return collected_dir
