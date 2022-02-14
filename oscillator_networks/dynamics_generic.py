import numpy as np
from scipy.integrate import ode, odeint, solve_ivp
from diffeqpy import de
from NumbaLSODA import lsoda_sig, lsoda
from numba import njit, cfunc

from dynamics_vectorfields import set_ode_jacobian, pointer_ode_vectorfield
from settings import STYLE_DYNAMICS_VALID, STYLE_DYNAMICS


def simulate_dynamics_general(init_cond, times, single_cell, dynamics_method="solve_ivp", **solver_kwargs):
    """
    single_cell is an instance of SingleCell
    See documentation on SciPy mehods here
    - https://docs.scipy.org/doc/scipy/reference/integrate.html
    """
    if dynamics_method == 'solve_ivp':
        r, times = ode_solve_ivp(init_cond, times, single_cell, **solver_kwargs)
    elif dynamics_method == "numba_lsoda":
        r, times = ode_numba_lsoda(init_cond, times, single_cell, **solver_kwargs)
    elif dynamics_method == "diffeqpy":
        r, times = ode_diffeqpy(init_cond, times, single_cell, **solver_kwargs)
    elif dynamics_method == "libcall":
        r, times = ode_libcall(init_cond, times, single_cell, **solver_kwargs)
    elif dynamics_method == "rk4":
        r, times = ode_rk4(init_cond, times, single_cell, **solver_kwargs)
    elif dynamics_method == "euler":
        r, times = ode_euler(init_cond, times, single_cell, **solver_kwargs)
    else:
        raise ValueError("method arg invalid, must be one of %s" % STYLE_DYNAMICS_VALID)
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


def ode_euler(init_cond, times, single_cell, **solver_kwargs):
    """
    single_cell is an instance of SingleCell
    """
    dt = times[1] - times[0]
    r = np.zeros((len(times), single_cell.graph_dim_ode))
    r[0] = np.array(init_cond)
    for idx, t in enumerate(times[:-1]):
        v = ode_system_vector(r[idx], None, single_cell)
        r[idx+1] = r[idx] + np.array(v)*dt
    return r, times


def ode_rk4(init_cond, times, single_cell, **solver_kwargs):
    """
    single_cell is an instance of SingleCell
    """
    dt = times[1] - times[0]
    r = np.zeros((len(times), single_cell.graph_dim_ode))
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


def ode_libcall(init_cond, times, single_cell, **solver_kwargs):
    """
    single_cell is an instance of SingleCell
    """
    fn = ode_system_vector
    r = odeint(fn, init_cond, times, args=(single_cell,), **solver_kwargs)
    return r, times


def ode_solve_ivp(init_cond, times, single_cell, **solver_kwargs):
    """
    single_cell is an instance of SingleCell
    method: see documentation here
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
    - stiff case: try 'Radau', 'BDF', 'LSODA'
    Note on solver_kwargs:
        solver_kwargs['dense_output'] = True
        solver_kwargs['t_eval'] = times or e.g. np.linspace(0, 50, 20000)
    """
    if 'atol' not in solver_kwargs.keys():
        solver_kwargs['atol'] = 1e-8
    if 'vectorized' not in solver_kwargs.keys():
        if single_cell.dim_ode > 1:
            solver_kwargs['vectorized'] = True
        else:
            solver_kwargs['vectorized'] = False
    fn = system_vector_obj_ode
    time_interval = [times[0], times[-1]]
    jac = None  #set_ode_jacobian(single_cell.style_ode)  # TODO investigate why singlecell traj much slower with Jacobian supplied

    # main solver call
    sol = solve_ivp(fn, time_interval, init_cond, args=(single_cell,), jac=jac, **solver_kwargs)

    r = np.transpose(sol.y)
    times = sol.t
    return r, times


def ode_diffeqpy(init_cond, times, single_cell, **solver_kwargs):
    """
    See documentation here: https://github.com/SciML/diffeqpy
    Also says to jit numba the vectorfield functions

    Function has the following form
    def f(u, p, t):
        return -u
    """
    fn = pointer_ode_vectorfield(single_cell.style_ode)
    time_interval = [times[0], times[-1]]

    # main solver call
    prob = de.ODEProblem(fn, init_cond, time_interval, single_cell.params_ode)
    sol = de.solve(prob, de.RadauIIA5(), **solver_kwargs)  # RadauIIA5() or lsoda(); can't call lsoda() from python?
    r = sol.u
    times = sol.t
    return r, times


def ode_numba_lsoda(init_cond, times, single_cell, **solver_kwargs):
    """ Docs: https://github.com/Nicholaswogan/NumbaLSODA
    @cfunc(lsoda_sig)
    def rhs(t, u, du, p):
        du[0] = u[0] - u[0] * u[1]
        du[1] = u[0] * u[1] - u[1] * p[0]

    Possible issue: our current arg format is not aligned
    def vectorfield_PWL3_swap(init_cond, params, t):

    We currently get this error:
    Untyped global name 'fn': Cannot determine Numba type of <class 'function'>
    File "dynamics_generic.py", line 163:
        def fn_mask(t, u, du, p):
            return fn(u, p, t)
            ^
    """
    fn = pointer_ode_vectorfield(single_cell.style_ode)

    @cfunc(lsoda_sig)
    def fn_mask(t, u, du, p):
        return fn(u, p, t)

    usol, success = lsoda(fn_mask, init_cond, times, data=single_cell.params_ode)
    r = usol.u
    times = usol.t
    return r, times
