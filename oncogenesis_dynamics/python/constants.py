"""
Comments
- current implementation for bifurcation along VALID_BIFURCATION_PARAMS only
- no stability calculation implemented (see matlab code for that)

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

PARAMS_DICT = {0: "bifurc_alpha_minus",
               1: "bifurc_alpha_plus",
               2: "bifurc_mu",
               3: "bifurc_a",
               4: "bifurc_b",
               5: "bifurc_c",
               6: "bifurc_N"}
VALID_BIFURCATION_PARAMS = ["bifurc_b"]  # list of implemented bifurcation parameters

X1_COL = "blue"  # blue stable (dashed unstable)
X2_COL = "green"  # green stable (dashed unstable)
