import csv
import matplotlib.pyplot as plt
import numpy as np
import os

from dynamics import simulate_dynamics_general
from settings import DYNAMICS_METHODS_VALID, DYNAMICS_METHOD, INIT_COND, TIME_START, TIME_END, NUM_STEPS
from vectorfields import set_params_ode, vectorfield_Yang2013


class SingleCell():

    def __init__(self, init_cond, label=''):
        """
        For numeric cell labels (network growth), use label='%d' % idx, for instance
        """
        self.dim_ode = 3           # dimension of ODE system
        self.dim_misc = 1          # dimension of misc. variables (e.g. fusome content)
        self.num_variables = self.dim_ode + self.dim_misc
        self.state_ode = init_cond
        self.style_ode = 'Yang2013'

        # make this flexible if other single cell ODEs are used
        assert self.style_ode == 'Yang2013'
        self.params_ode = set_params_ode()

        # setup names for all dynamical variables
        self.variables_short = {0: 'x',
                                1: 'y',
                                2: 'z',
                                3: 'f'}
        self.variables_long = {0: 'Cyclin active',
                               1: 'Cyclin total',
                               2: 'Modulator, e.g. Bam',
                               3: 'Fusome content'}
        if label != '':
            for idx in range(self.num_variables):
                self.variables_short[idx] += '_%s' % label
                self.variables_long[idx] += ' (Cell %s)' % label

    def ode_system_vector(self, init_cond):
        p = self.params_ode  # TODO if there is feedback these 'constants' might be pseudo-dynamic [see xyz params.py]
        x, y, z = init_cond
        if self.style_ode == 'Yang2013':
            vectorfield = vectorfield_Yang2013(self.params_ode, x, y, z=z, two_dim=False)
        else:
            print('Error: invalid self.style_ode', self.style_ode)
            assert self.style_ode == 'Yang2013'
            vectorfield = None

        return vectorfield

    def trajectory(self, init_cond=None, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS,
                   dynamics_method=DYNAMICS_METHOD, flag_info=False):
        if init_cond is None:
            init_cond = self.state_ode

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
    sc = SingleCell(init_cond, label='foo')
    r, times = sc.trajectory(flag_info=True, dynamics_method='libcall')
    print(r, times)
    print(r.shape)

    plt.plot(times, r, label=['x', 'y', 'z'])
    plt.xlabel(r'$t$ [min]')
    plt.ylabel(r'concentration [nM]')
    plt.legend()
    plt.show()
