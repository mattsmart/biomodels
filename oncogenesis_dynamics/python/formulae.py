# FUNCTIONS
def bifurc_get(bifurc_name):
    # assumes threshold_2 is stronger constraint, atm hardcode rearrange expression for bifruc param
    if bifurc_name == "delta":
        bifurc_val = alpha_minus * alpha_plus / (s + alpha_plus) - (s + alpha_minus + mu)
        return bifurc_val
    else:
        raise ValueError(bifruc_name + ' not valid bifurc_name')

def threshold_1(delta):
    return 2 * s + delta + alpha_plus + alpha_minus + mu


def threshold_2(delta):
    return (s + alpha_plus) * (s + delta + alpha_minus + mu) - alpha_minus * alpha_plus


def q_get(sign, delta):
    assert sign in [-1, +1]
    bterm = alpha_plus - alpha_minus - mu - delta
    return 0.5 / alpha_minus * (bterm + sign * np.sqrt(bterm ** 2 + 4 * alpha_minus * alpha_plus))

def xvec_get(q, delta):
    xi = N * (s + alpha_plus - alpha_minus * q) / (s + (delta + s) * q)
    yi = q * xi
    zi = N - xi - yi
    return xi, yi, zi