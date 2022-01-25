import numpy as np

from settings import VALID_STYLE_ODE


def set_ode_attributes(style_ode):
    dim_ode = 3  # dimension of ODE system
    dim_misc = 2  # dimension of misc. variables (e.g. fusome content)
    variables_short = {0: 'Cyc_act',
                       1: 'Cyc_tot',
                       2: 'Bam',
                       3: 'n_div',
                       4: 'fusome'}
    variables_long = {0: 'Cyclin active',
                      1: 'Cyclin total',
                      2: 'Modulator, e.g. Bam',
                      3: 'Number of Divisions',
                      4: 'Fusome content'}
    if style_ode == 'Yang2013':
        # currently, all methods share the same attributes above
        pass
    elif style_ode == 'PWL2':
        # currently, all methods share the same attributes above
        pass
    elif style_ode == 'toy_flow':
        dim_ode = 1  # dimension of ODE system
        dim_misc = 0  # dimension of misc. variables (e.g. fusome content)
        variables_short = {0: 'vol'}
        variables_long = {0: 'Water Volume'}
    else:
        print("Warning: style_ode %s is not supported by set_ode_attributes()" % style_ode)
        print("Supported odes include:", VALID_STYLE_ODE)
    assert len(variables_short.keys()) == len(variables_long.keys())
    assert len(variables_short.keys()) == (dim_ode + dim_misc)
    return dim_ode, dim_misc, variables_short, variables_long


def set_ode_params(style_ode):
    if style_ode == 'Yang2013':
        # reference is Yang2013 Table S1
        p = {
            'k_synth': 1,  # nM / min
            'a_deg': 0.01,  # min^-1
            'b_deg': 0.04,  # min^-1
            'EC50_deg': 32,  # nM
            'n_deg': 17,  # unitless
            'a_Cdc25': 0.16,  # min^-1
            'b_Cdc25': 0.80,  # min^-1
            'EC50_Cdc25': 35,  # nM
            'n_Cdc25': 11,  # unitless
            'a_Wee1': 0.08,  # min^-1
            'b_Wee1': 0.40,  # min^-1
            'EC50_Wee1': 30,  # nM
            'n_Wee1': 3.5,  # unitless
        }
        # add any extra parameters that are separate from Yang2013
        p['Bam_activity'] = 1  # as indicated in SmallCellCluster review draft p7
        p['Bam_deg'] = 0  # degradation rate; arbitrary, try 0 or 1e-2 to 1e-4
    elif style_ode == 'PWL2':
        """ Notes from Hayden slide 12:
        - ((1−𝛾))/2𝜀𝛾 is duration that green intersects red between extrema of red
        - the free params are 𝑎, 𝐼, 𝜀𝛾/(1+𝛾), b
        - Maybe specify conditions on 𝛾
        """
        p = {
            'C': 1e-1,              # speed scale for fast variable Cyc_act
            'a': 2,                 # defines the corners of PWL function for x
            'b': 2,                 # defines the y-intercept of the dy/dt=0 nullcline, y(x) = 1/gamma * (-x + b)
            'gamma': 1e-1,          # degradation of Cyc_tot
            'epsilon': 0.3,         # rate of inhibitor accumulation
            'I_initial': 0,         # initial inhibitor (e.g. Bam) concentration
            't_pulse_switch': 25.0  # treating inhibitor timeseries as pulse with a negative slope from t=T to t=2T
        }
        assert 0 < p['C'] < 1
        assert 0 < p['gamma']
        assert 0 <= p['epsilon']
    elif style_ode == 'toy_flow':
        p = {}
    else:
        print("Warning: style_ode %s is not supported by get_params_ODE()" % style_ode)
        print("Supported odes include:", VALID_STYLE_ODE)
        p = {}
    return p


def set_ode_vectorfield(style_ode, params, init_cond, two_dim=True, **ode_kwargs):
    if style_ode == 'Yang2013':
        dxdt = vectorfield_Yang2013(params, init_cond, z=ode_kwargs.get('z', 0), two_dim=two_dim)
    elif style_ode == 'PWL2':
        dxdt = vectorfield_PWL2(params, init_cond, ode_kwargs.get('t', 0), z=ode_kwargs.get('z', 0), two_dim=two_dim)
    elif style_ode == 'toy_flow':
        dxdt = vectorfield_toy()
    else:
        print("Warning: style_ode %s is not supported by set_ode_vectorfield()" % style_ode)
        print("Supported odes include:", VALID_STYLE_ODE)
        dxdt = None
    return dxdt


def ode_integration_defaults(style_ode):
    t0 = 0.0
    if style_ode == 'Yang2013':
        t1 = 800
        num_steps = 2000
        init_cond = [60.0, 0.0, 0.0]
    elif style_ode == 'PWL2':
        t1 = 50
        num_steps = 2000
        init_cond = [1.0, 1.0, 0.0]
    elif style_ode == 'toy_flow':
        t1 = 50
        num_steps = 2000
        init_cond = [100.0]
    else:
        print("Warning: style_ode %s is not supported by ode_integration_defaults()" % style_ode)
        print("Supported odes include:", VALID_STYLE_ODE)
        t1 = None
        num_steps = None
        init_cond = None
    return t0, t1, num_steps, init_cond


def vectorfield_toy():
    return [0]


def vectorfield_Yang2013(params, init_cond, z=0, two_dim=True):
    """
    Args:
        params - dictionary of ODE parameters used by Yang2013
        x - array-like
        y - array-like
        z - array-like
    Returns:
        array like of shape [x, y] or [x, y, z] depending on two_dim flag
    """
    p = params
    x, y, _ = init_cond  # TODO note z is passed through init_cond but is unused; use static "external" z for now

    # "f(x)" factor of the review - degradation
    # TODO care if x = 0 -- add case?
    r_degradation = (p['EC50_deg'] / x) ** p['n_deg']
    degradation = p['a_deg'] + p['b_deg'] / (1 + r_degradation)
    degradation_scaled = degradation / (1 + z / p['Bam_activity'])  # as in p7 of SmallCellCluster Review draft

    # "g(x)" factor of the review - activation by Cdc25
    # TODO care if x = 0 -- add case?
    r_activation = (p['EC50_Cdc25'] / x) ** p['n_Cdc25']
    activation = p['a_Cdc25'] + p['b_Cdc25'] / (1 + r_activation)

    # "k_i" factor of the review - de-activation by Wee1
    r_deactivation_inv = (x / p['EC50_Wee1']) ** p['n_Wee1']
    deactivation = p['a_Wee1'] + p['b_Wee1'] / (1 + r_deactivation_inv)

    dxdt = p['k_synth'] - degradation_scaled * x + activation * (y - x) - deactivation * x
    dydt = p['k_synth'] - degradation_scaled * y
    #dzdt = np.zeros_like(dxdt)
    dzdt = -p['Bam_deg'] * z

    if two_dim:
        out = [dxdt, dydt]
    else:
        out = [dxdt, dydt, dzdt]

    return out


def PWL_g_of_x_SCALAR(params, x):
    """
    Currently unused; see vectorized variant PWL_g_of_x()
    """
    a = params['a']
    if x < (a/2):
        g = -x
    elif x <= ((1+a)/2):
        g = x - a
    else:
        g = 1 - x
    return g


def PWL_g_of_x(params, x):
    a = params['a']
    g1 = np.where(x < a/2, x, 0)
    g2 = np.where(
        ((a/2) <= x) & (x < ((1+a)/2)),
        -x + a, 0)
    g3 = np.where(x >= ((1+a)/2), -1 + x, 0)
    g = g1 + g2 + g3
    return g


def PWL_I_of_t_pulse(params, t):
    """
    Generates a triangular pulse rising at t=0 with switch at t = params['t_pulse_switch']
      when t > 2 * params['t_pulse_switch'], there is no further change
    """
    if t < params['t_pulse_switch']:
        I = params['I_initial'] + params['epsilon'] * t
    elif t < 2 * params['t_pulse_switch']:
        I = params['I_initial'] - params['epsilon'] * t + 2 * params['epsilon'] * params['t_pulse_switch']
    else:
        I = params['I_initial']
    assert I >= 0
    assert t >= 0
    return I


def vectorfield_PWL2(params, init_cond, t, z=0, two_dim=True):
    """
    Originally from slide 12 of Hayden ppt
    - Change #1: here the variables are relabelled (based on Jan 18 discussion)
        x = -1 * v
        y = w
        - note x only degrades in the intermediate regime of PWL g(x) now, because of the relabelling
    - Change #2:
        Need to shift the y-nullcline to the right in order to have positive stable states
        Use new parameter "b" which is basal rate of y production
    Args:
        params - dictionary of ODE parameters used by piecewise linear ODE system
        x - array-like
        y - array-like
        z - array-like
        t - time corresponding to integration variable (non-autonomous system)
    Returns:
        array like of shape [x, y] or [x, y, z] depending on two_dim flag
    """
    x, y, _ = init_cond  # TODO note z is passed through init_cond but is unused; use static "external" z for now

    I_of_t = PWL_I_of_t_pulse(params, t)
    g_of_x = PWL_g_of_x(params, x)

    dxdt = 1/params['C'] * (y - g_of_x - I_of_t)
    dydt = params['b'] - x - params['gamma'] * y
    dzdt = np.zeros_like(dxdt)
    #dzdt = -p['Bam_deg'] * z

    if two_dim:
        out = [dxdt, dydt]
    else:
        out = [dxdt, dydt, dzdt]

    return out
