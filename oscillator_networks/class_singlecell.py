import csv
import matplotlib.pyplot as plt
import numpy as np
import os

from dynamics_generic import simulate_dynamics_general
from dynamics_vectorfields import set_ode_params, ode_integration_defaults, set_ode_vectorfield, set_ode_attributes
from file_io import run_subdir_setup
from settings import STYLE_DYNAMICS, STYLE_ODE


class SingleCell():

    def __init__(self, init_cond_ode=None, style_ode=STYLE_ODE, params_ode=None, label=''):
        """
        For numeric cell labels (network growth), use label='%d' % idx, for instance
        """
        self.style_ode = style_ode
        dim_ode, dim_misc, variables_short, variables_long = set_ode_attributes(style_ode)
        self.dim_ode = dim_ode            # dimension of ODE system
        self.dim_misc = dim_misc          # dimension of misc. variables (e.g. fusome content)
        self.num_variables = self.dim_ode + self.dim_misc
        # setup names for all dynamical variables
        self.variables_short = variables_short
        self.variables_long = variables_long
        if label != '':
            for idx in range(self.num_variables):
                self.variables_short[idx] += '_%s' % label
                self.variables_long[idx] += ' (Cell %s)' % label
        # make this flexible if other single cell ODEs are used
        self.state_ode = init_cond_ode
        self.params_ode = params_ode
        if self.params_ode is None:
            self.params_ode = set_ode_params(self.style_ode)

    def ode_system_vector(self, init_cond, t):
        p = self.params_ode  # TODO if there is feedback these 'constants' might be pseudo-dynamic [see xyz params.py]
        ode_kwargs = {
            'z': init_cond[-1],  # TODO remove this hacky way to pass z its not safe (handling PWL, Yang2013 here might be slow though)
            't': t
        }
        dxdt = set_ode_vectorfield(self.style_ode, self.params_ode, init_cond, **ode_kwargs)
        return dxdt

    def trajectory(self, init_cond=None, t0=None, t1=None, num_steps=None, dynamics_method=STYLE_DYNAMICS,
                   flag_info=False, **solver_kwargs):
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

        r, times = simulate_dynamics_general(init_cond, times, self, method=dynamics_method, **solver_kwargs)
        if flag_info:
            print('Done trajectory\n')

        return r, times

    def printer(self):
        print('dim_ode:', self.dim_ode)
        print('dim_misc:', self.dim_misc)
        print('num_variables:', self.num_variables)
        print("State variables:")
        for idx in range(self.num_variables):
            print("\t %s: %s | %s" % (idx, self.variables_short[idx], self.variables_long[idx]))

    def write_ode_params(self, fpath):
        with open(fpath, "a", newline='\n') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            keys = list(self.params_ode.keys())
            keys.sort()
            for k in keys:
                writer.writerow([k, self.params_ode[k]])


if __name__ == '__main__':
    style_ode = 'PWL3_swap'  # PWL4_auto_linear
    sc = SingleCell(label='c1', style_ode=style_ode)
    if style_ode in ['PWL2', 'PWL3', 'PWL3_swap']:
        sc.params_ode['C'] = 1e-2
        sc.params_ode['epsilon'] = 0.1
        sc.params_ode['t_pulse_switch'] = 25
    if style_ode in ['PWL4_auto_linear']:
        sc.params_ode['a'] = 2
        sc.params_ode['d'] = 1

    solver_kwargs = {
        'atol': 1e-8,
        'dense_output': False,  # seems to have no effect
        't_eval': None} #np.linspace(0, 100, 2000)}
    r, times = sc.trajectory(flag_info=True, dynamics_method='solve_ivp', **solver_kwargs)

    io_dict = run_subdir_setup(run_subfolder='singlecell')

    print(r.shape)
    np.savetxt(io_dict['basedir'] + os.sep + 'traj_times.txt', times)
    np.savetxt(io_dict['basedir'] + os.sep + 'traj_x.txt', r)

    plt.plot(times, r, label=[sc.variables_short[i] for i in range(sc.dim_ode)])
    plt.xlabel(r'$t$ [min]')
    plt.ylabel(r'concentration [nM]')
    plt.legend()
    plt.savefig(io_dict['basedir'] + os.sep + 'traj_example.jpg')
    plt.show()
