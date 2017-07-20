"""
Conventions
- params is 7-vector of the form: params[0] -> alpha_plus
                                  params[1] -> alpha_minus
                                  params[2] -> mu
                                  params[3] -> a           (usually normalized to 1)
                                  params[4] -> b           (b = 1 - delta)
                                  params[5] -> c           (c = 1 + s)
                                  params[6] -> N
- if an element of params is specified as None then a bifurcation range will be be found and used
"""


def bifurc_value(params, bifurc_name):
    """
    Note: assumes params contains at most one None parameter
    """
    alpha_plus = params[0]
    alpha_minus = params[1]
    mu = params[2]
    a = params[3]
    b = params[4]
    c = params[5]
    N = params[6]
    if b is not None:
        delta = 1 - b
    if c is not None:
        s = c - 1
    # assumes threshold_2 is stronger constraint, atm hardcode rearrange expression for bifruc param
    if bifurc_name == "bifurc_b":
        bifurc_val = alpha_minus * alpha_plus / (s + alpha_plus) - (s + alpha_minus + mu)
        return bifurc_val
    else:
        raise ValueError(bifurc_name + ' not valid bifurc ID')


def threshold_1(delta):
    return 2 * s + delta + alpha_plus + alpha_minus + mu


def threshold_2(delta):
    return (s + alpha_plus) * (s + delta + alpha_minus + mu) - alpha_minus * alpha_plus


def q_get(sign, delta):
    assert sign in [-1, +1]
    bterm = alpha_plus - alpha_minus - mu - delta
    return 0.5 / alpha_minus * (bterm + sign * np.sqrt(bterm ** 2 + 4 * alpha_minus * alpha_plus))


def fp_location(q, delta):
    xi = N * (s + alpha_plus - alpha_minus * q) / (s + (delta + s) * q)
    yi = q * xi
    zi = N - xi - yi
    return xi, yi, zi