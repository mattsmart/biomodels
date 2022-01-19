import csv
import matplotlib.pyplot as plt
import numpy as np
import os

from dynamics_generic import simulate_dynamics_general
from dynamics_vectorfields import ode_choose_params, ode_integration_defaults, ode_choose_vectorfield
from file_io import run_subdir_setup
from settings import DYNAMICS_METHOD, STYLE_ODE


class SingleCell():

    def __init__(self, init_cond, style_ode=STYLE_ODE, params_ode=None, label=''):
        """
        For numeric cell labels (network growth), use label='%d' % idx, for instance
        """
        self.dim_ode = 3           # dimension of ODE system
        self.dim_misc = 2          # dimension of misc. variables (e.g. fusome content)
        self.num_variables = self.dim_ode + self.dim_misc
        self.state_ode = init_cond
        self.style_ode = style_ode
        self.params_ode = params_ode

        # make this flexible if other single cell ODEs are used
        if self.params_ode is None:
            self.params_ode = ode_choose_params(self.style_ode)

        # setup names for all dynamical variables
        self.variables_short = {0: 'Cyc_act',
                                1: 'Cyc_tot',
                                2: 'Bam',
                                3: 'ndiv',
                                4: 'fusome'}
        self.variables_long = {0: 'Cyclin active',
                               1: 'Cyclin total',
                               2: 'Modulator, e.g. Bam',
                               3: 'Number of Divisions',
                               4: 'Fusome content'}
        if label != '':
            for idx in range(self.num_variables):
                self.variables_short[idx] += '_%s' % label
                self.variables_long[idx] += ' (Cell %s)' % label

    def ode_system_vector(self, init_cond, t):
        p = self.params_ode  # TODO if there is feedback these 'constants' might be pseudo-dynamic [see xyz params.py]
        x, y, z = init_cond
        ode_kwargs = {
            'z': z,
            't': t
        }
        dxdt = ode_choose_vectorfield(self.style_ode, self.params_ode, x, y, two_dim=False, **ode_kwargs)
        return dxdt

    def trajectory(self, init_cond=None, t0=None, t1=None, num_steps=None, dynamics_method=DYNAMICS_METHOD, flag_info=False):
        # integration parameters
        T0, T1, NUM_STEPS, INIT_COND = ode_integration_defaults(self.style_ode)
        if init_cond is None:
            init_cond = self.state_ode
            if self.state_ode is None:
                init_cond = INIT_COND
        if t0 is None:
            t0 = T0
        if t1 is None:
            t1 = T1
        if num_steps is None:
            num_steps = NUM_STEPS
        times = np.linspace(t0, t1, num_steps + 1)
        if flag_info:
            times = np.linspace(t0, t1, num_steps + 1)
            print("ODE Setup: t0, t1:", t0, t1, "| num_steps, dt:", num_steps, times[1] - times[0])
            print("Init Cond:", init_cond)
            self.printer()

        r, times = simulate_dynamics_general(init_cond, times, self, method=dynamics_method)
        if flag_info:
            print('Done trajectory\n')

        return r, times

    def printer(self):
        print("State variables:")
        for idx in range(self.num_variables):
            print("\t %s: %s | %s" % (idx, self.variables_short[idx], self.variables_long[idx]))

    def io_write(self, filedir, filename):
        # TODO
        """
        filepath = filedir + os.sep + filename
        with open(filepath, "wb") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for idx in range(len(PARAMS_ID.keys())):
                val = self.params_list[idx]
                if self.params_list[idx] is None:
                    val = 'None'
                writer.writerow([PARAMS_ID[idx], val])
            # any extra non-dynamics params
            writer.writerow(['system', self.system])
            writer.writerow(['feedback', self.feedback])"""
        return filepath


if __name__ == '__main__':
    init_cond = (60.0, 0.0, 0.0)
    sc = SingleCell(init_cond, label='c1', style_ode='Yang2013')
    r, times = sc.trajectory(flag_info=True, dynamics_method='solve_ivp')
    print(r, times)
    print(r.shape)

    io_dict = run_subdir_setup()

    plt.plot(times, r, label=[sc.variables_short[0], sc.variables_short[1], sc.variables_short[2]])
    plt.xlabel(r'$t$ [min]')
    plt.ylabel(r'concentration [nM]')
    plt.legend()
    plt.savefig(io_dict['basedir'] + os.sep + 'traj_example.jpg')
    plt.show()