import csv
import numpy as np
import os


class SingleCell():

    def __init__(self, init_cond, label=''):
        """
        For numeric cell labels (network growth), use label='%d' % idx, for instance
        """
        self.num_ode = 3           # dimension of ODE system
        self.num_variables = 4     # dimension of total state space (ODE + misc.)
        self.state = init_cond

        # setup names for all dynamical variables
        self.variables_short = {0: 'x',
                                1: 'y',
                                2: 'z',
                                3: 'f'}
        self.variables_long = {0: 'Oscillator A - Cyclin active',
                               1: 'Oscillator B - Cyclin total',
                               2: 'Modulator, e.g. Bam',
                               3: 'Fusome content'}
        if label != '':
            for idx in range(self.num_variables):
                self.variables_short[idx] += '_%s' % label
                self.variables_long[idx] += ' (Cell %s)' % label

s
    def ode_system_vector(self, init_cond):
        p = self  # if there is feedback these 'constants' might be pseudo-dynamic -- see xyz params.py
        x, y, z = init_cond
        dxdt = 0
        dydt = 0
        dzdt = 0
        """
        dxdt = p.v_x - x * (p.alpha_plus + p.mu_base) + y * p.alpha_minus + (p.a - fbar) * x
        dydt = p.v_y + x * p.alpha_plus - y * (p.alpha_minus + p.mu) + (p.b - fbar) * y
        dzdt = p.v_z + y * p.mu + x * p.mu_base + (p.c - fbar) * z
        """
        return [dxdt, dydt, dzdt]

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
