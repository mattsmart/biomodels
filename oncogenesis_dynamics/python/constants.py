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
                                  params[6] -> N           (float not int)
                                  params[7] -> v_x
                                  params[8] -> v_y         (typically 0)
                                  params[9] -> v_z         (typically 0)
- if an element of params is specified as None then a bifurcation range will be be found and used
"""

PARAMS_ID = {0: "alpha_minus",
             1: "alpha_plus",
             2: "mu",
             3: "a",
             4: "b",
             5: "c",
             6: "N",
             7: "v_x",
             8: "v_y",
             9: "v_z"}
BIFURC_DICT = {i: "bifurc_" + PARAMS_ID[i] for i in xrange(len(PARAMS_ID))}
VALID_BIFURC_PARAMS = ["bifurc_b"]  # list of implemented bifurcation parameters

OUTPUT_DIR = "output"

X1_COL = "blue"  # blue stable (dashed unstable)
X2_COL = "green"  # green stable (dashed unstable)
