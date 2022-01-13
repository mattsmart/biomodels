import csv
import matplotlib.pyplot as plt
import numpy as np
import os

from dynamics import simulate_dynamics_general
from settings import DYNAMICS_METHODS_VALID, DYNAMICS_METHOD, INIT_COND, TIME_START, TIME_END, NUM_STEPS


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
        # reference is Yang2013 Table S1
        self.params_ode = {
            'k_synth': 1,      # nM / min
            'a_deg': 0.01,     # min^-1
            'b_deg': 0.04,     # min^-1
            'EC50_deg': 32,    # nM
            'n_deg': 17,       # unitless
            'a_Cdc25': 0.16,   # min^-1
            'b_Cdc25': 0.80,   # min^-1
            'EC50_Cdc25': 35,  # nM
            'n_Cdc25': 11,     # unitless
            'a_Wee1': 0.08,    # min^-1
            'b_Wee1': 0.40,    # min^-1
            'EC50_Wee1': 30,   # nM
            'n_Wee1': 3.5,     # unitless
        }

        # add any extra parameters
        self.params_ode['k_Bam'] = 1  # as indicated in SmallCellCluster review draft p7

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
        p = self.params_ode  # if there is feedback these 'constants' might be pseudo-dynamic [see xyz params.py]
        x, y, z = init_cond
        if self.style_ode == 'Yang2013':
            # setup factors
            k_synth = p['k_synth']

            # "f(x)" factor of the review
            x_d = x ** p['n_deg']
            ec50_d = p['EC50_deg'] ** p['n_deg']
            degradation = p['a_deg'] + p['b_deg'] * x_d / (ec50_d + x_d)
            degradation_scaled = degradation / (1 + z / p['k_Bam'])

            # "g(x)" factor of the review - activation by Cdc25
            x_plus = x ** p['n_Cdc25']
            ec50_plus = p['EC50_Cdc25'] ** p['n_Cdc25']
            activation = p['a_Cdc25'] + p['b_Cdc25'] * x_plus / (ec50_plus + x_plus)

            # "k_i" factor of the review - de-activation by Wee1
            x_minus = x ** p['n_Wee1']
            ec50_minus = p['EC50_Wee1'] ** p['n_Wee1']
            deactivation = p['a_Wee1'] + p['b_Wee1'] * ec50_minus / (ec50_minus + x_minus)

            dxdt = k_synth - degradation_scaled * x + activation * (x-y) - deactivation * x
            dydt = k_synth - degradation_scaled * y
            dzdt = 0  # TODO keep constant inhibitor for now (see review draft)
        else:
            print('Error: invalid self.style_ode', self.style_ode)
            assert self.style_ode == 'Yang2013'
        """
        dxdt = p.v_x - x * (p.alpha_plus + p.mu_base) + y * p.alpha_minus + (p.a - fbar) * x
        dydt = p.v_y + x * p.alpha_plus - y * (p.alpha_minus + p.mu) + (p.b - fbar) * y
        dzdt = p.v_z + y * p.mu + x * p.mu_base + (p.c - fbar) * z
        """
        return [dxdt, dydt, dzdt]

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
    init_cond = (1, 2, 0)
    sc = SingleCell(init_cond, label='foo')
    r, times = sc.trajectory(flag_info=True, dynamics_method='libcall')
    print(r, times)
    print(r.shape)

    plt.plot(times, r, label=['x', 'y', 'z'])
    plt.xlabel(r'$t$ [min]')
    plt.ylabel(r'concentration [nM]')
    plt.legend()
    plt.show()
