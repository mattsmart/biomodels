import numpy as np
from scipy.integrate import ode, odeint, solve_ivp

from settings import DYNAMICS_METHODS_VALID, DYNAMICS_METHOD


def simulate_dynamics_general(init_cond, times, single_cell, method="solve_ivp"):
    """
    single_cell is an instance of SingleCell
    See documentation on SciPy mehods here
    - https://docs.scipy.org/doc/scipy/reference/integrate.html
    """
    if method == 'solve_ivp':
        r, times = ode_solve_ivp(init_cond, times, single_cell)
    elif method == "libcall":
        r, times = ode_libcall(init_cond, times, single_cell)
    elif method == "rk4":
        r, times = ode_rk4(init_cond, times, single_cell)
    elif method == "euler":
        r, times = ode_euler(init_cond, times, single_cell)
    else:
        raise ValueError("method arg invalid, must be one of %s" % DYNAMICS_METHODS_VALID)
    """ TODO consider implemeneting:
    elif method == "gillespie":
        return stoch_gillespie(init_cond, len(times), params)
    elif method == "bnb":
        return stoch_bnb(init_cond, len(times), params)
    elif method == "tauleap":
        return stoch_tauleap(init_cond, len(times), params)
    """
    return r, times


def ode_system_vector(init_cond, t, single_cell):
    """
    Wrapper used by ode_libcall [format is: traj = odeint(fn, x0, t, args=(b, c))]
    - single_cell is an instance of SingleCell
    """
    dxvec_dt = single_cell.ode_system_vector(init_cond, t)
    return dxvec_dt


def system_vector_obj_ode(t_scalar, r_idx, single_cell):
    """
    Wrapper used by ode_rk4
    - single_cell is an instance of SingleCell
    """
    return ode_system_vector(r_idx, t_scalar, single_cell)


def ode_euler(init_cond, times, single_cell):
    """
    single_cell is an instance of SingleCell
    """
    dt = times[1] - times[0]
    r = np.zeros((len(times), single_cell.dim_ode))
    r[0] = np.array(init_cond)
    for idx, t in enumerate(times[:-1]):
        v = ode_system_vector(r[idx], None, single_cell)
        r[idx+1] = r[idx] + np.array(v)*dt
    return r, times


def ode_rk4(init_cond, times, single_cell):
    """
    single_cell is an instance of SingleCell
    """
    dt = times[1] - times[0]
    r = np.zeros((len(times), single_cell.dim_ode))
    r[0] = np.array(init_cond)
    obj_ode = ode(system_vector_obj_ode, jac=None)
    obj_ode.set_initial_value(init_cond, times[0])
    obj_ode.set_f_params(single_cell)
    obj_ode.set_integrator('dopri5')
    idx = 1
    while obj_ode.successful() and obj_ode.t < times[-1]:
        obj_ode.integrate(obj_ode.t + dt)
        r[idx] = np.array(obj_ode.y)
        idx += 1
    return r, times


def ode_libcall(init_cond, times, single_cell):
    """
    single_cell is an instance of SingleCell
    """
    fn = ode_system_vector
    r = odeint(fn, init_cond, times, args=(single_cell,))
    return r, times


def ode_solve_ivp(init_cond, times, single_cell, method='Radau'):
    """
    single_cell is an instance of SingleCell
    method: see documentation here
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
    - stiff case: try 'Radau', 'BDF', 'LSODA'
    """
    fn = system_vector_obj_ode
    time_interval = [times[0], times[-1]]
    sol = solve_ivp(fn, time_interval, init_cond, method=method, vectorized=True, args=(single_cell,))
    r = np.transpose(sol.y)
    times = sol.t
    return r, times