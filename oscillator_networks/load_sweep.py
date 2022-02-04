import os


def analyze_sweep_1D(dir_sweep, param_name):

    for subdir in dir_sweep:

        # A) load classdump

        # B) extract useful info and add to collection

    # Visualize info

    return


if __name__ == '__main__':
    dir_sweep = 'runs' + os.sep + 'sweep_1d_epsilon_0.0_0.5_20'
    param_name = 'epsilon'
    analyze_sweep_1D(dir_sweep, param_name)
