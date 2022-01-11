import csv
import numpy as np
import os


class SingleCell():

    def __init__(self, init_cond, label=''):
        """
        For numeric cell labels (network growth), use label='%d' % idx, for instance
        """
        self.num_variables = 4
        if label != '':
            label_short = '_%s' % label
            label_long = '(Cell %s)' % label
        self.variables_short = {0: 'x%s' % label_short,
                                1: 'y%s' % label_short,
                                2: 'z%s' % label_short,
                                3: 'f%s' % label_short}
        self.variables_long = {0: 'Oscillator A - Cyclin active ' % label_long,
                               1: 'Oscillator B - Cyclin total' % label_long,
                               2: 'Modulator, e.g. Bam' % label_long,
                               3: 'Fusome content' % label_long}
        self.state = init_cond

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
