import csv
from os import sep

from constants import ODE_SYSTEMS, PARAMS_ID, PARAMS_ID_INV, PARAM_Z0_RATIO, PARAM_Y0_PLUS_Z0_RATIO, PARAM_HILL, \
                      PARAMS_ID_INV, PARAM_GAMMA, DEFAULT_FEEDBACK_SHAPE, FEEDBACK_SHAPES


class Params(object):

    def __init__(self, params_dict, system, init_cond=None, feedback=DEFAULT_FEEDBACK_SHAPE):
        # TODO maybe have attribute self.params_id which is dict int->param_name (diff for diff system)
        # TODO two system params, one if there is feedback, other is feedback type (i.e. hill n=1, step with threshold)
        # TODO have "default params" corresponding to specific models (e.g xyz default with alpha_minus=0 is like 2-hit)
        # params_dict as scalars #TODO could set these to defaults but this way catches bugs better maybe
        self.alpha_plus = None
        self.alpha_minus = None
        self.mu = None
        self.a = None
        self.b = None
        self.c = None
        self.c2 = None
        self.N = None
        self.v_x = None
        self.v_y = None
        self.v_z = None
        self.v_z2 = None
        self.mu_base = None
        for k, v in params_dict.iteritems():
            setattr(self, k, v)
        # store params_dict as canonical list and add the Nones to self.params_dict
        self.params_list = [getattr(self, PARAMS_ID[idx]) for idx in xrange(len(PARAMS_ID.keys()))]
        # add the leftover elements to params_dict
        self.params_dict = params_dict
        keys_to_add = set(PARAMS_ID.values()) - set(params_dict.keys())
        for key in keys_to_add:
            self.params_dict[key] = getattr(self, key)
        # init_cond as x, y, z, etc
        self.init_cond = init_cond  # TODO not fully implemented
        # system as defined in constants.pu (e.g. 'default', 'feedback_z')
        assert system in ODE_SYSTEMS
        assert feedback in FEEDBACK_SHAPES
        self.system = system
        self.feedback = feedback
        if system == "default":
            assert self.mu_base == 0.0
            assert self.feedback == "constant"
        if system in ["default", "feedback_z", "feedback_yz"]:
            self.numstates = 3
            self.states = {0: "x", 1: "y", 2: "z"}           # can also do FPT to a 4th state z2
            self.growthrates = [self.a, self.b, self.c]
            self.flowrates = [self.v_x, self.v_y, self.v_z]
            self.constant_growthandflowrates = True
            self.update_dict = {
                 0: [1, 0, 0], 1: [-1, 0, 0],                  # birth/death events for x
                 2: [0, 1, 0], 3: [0, -1, 0],                  # birth/death events for y
                 4: [0, 0, 1], 5: [0, 0, -1],                  # birth/death events for z
                 6: [-1, 1, 0], 7: [1, -1, 0], 8: [0, -1, 1],  # transition events
                 9: [1, 0, 0], 10: [0, 1, 0], 11: [0, 0, 1],   # immigration events
                 12: [-1, 0, 1], 13: [0, 0, -1]}               # special x->z, fpt z1->z2 (z2 untracked) transitions
        elif system in ["feedback_mu_XZ_model"]:
            assert self.mu_base > 0
            self.numstates = 2
            self.states = {0: "x", 1: "z"}                   # can also do FPT to a 3rd state z2
            self.growthrates = [self.a, self.c]
            self.flowrates = [self.v_x, self.v_z]
            self.b = 0.0             # TODO make None and optimize truncated eqns
            self.v_y = 0.0           # TODO make None and optimize truncated eqns
            self.alpha_plus = 0.0    # TODO make None and optimize truncated eqns
            self.alpha_minus = 0.0   # TODO make None and optimize truncated eqns
            self.constant_growthandflowrates = True
            self.update_dict = {
                 0: [1, 0], 1: [-1, 0],             # birth/death events for x
                 2: [0, 1], 3: [0, -1],             # birth/death events for z
                 4: [-1, 1],                        # transition events (x to z mutation)
                 5: [1, 0], 6: [0, 1],              # immigration events
                 7: [0, -1]}                        # special first-passage event z1->z2 (z2 untracked)
        elif system in ["feedback_XYZZprime"]:
            assert self.v_z2 is not None
            assert self.c2 is not None
            self.numstates = 4
            self.states = {0: "x", 1: "y", 2: "z", 3: "z2"}  # can also do FPT to a 5th state z3
            self.growthrates = [self.a, self.b, self.c, self.c2]
            self.flowrates = [self.v_x, self.v_y, self.v_z, self.v_z2]
            self.constant_growthandflowrates = True
            self.update_dict = {
                0: [1, 0, 0, 0], 1: [-1, 0, 0, 0],     # birth/death events for x
                2: [0, 1, 0, 0], 3: [0, -1, 0, 0],     # birth/death events for y
                4: [0, 0, 1, 0], 5: [0, 0, -1, 0],     # birth/death events for z
                6: [0, 0, 0, 1], 7: [0, 0, 0, -1],     # birth/death events for z2
                8: [-1, 1, 0, 0], 9: [1, -1, 0, 0],    # transition events (xy, yx)
                10: [0, -1, 1, 0], 11: [0, 0, -1, 1],  # transition events (yz, zz2)
                12: [1, 0, 0, 0], 13: [0, 1, 0, 0], 14: [0, 0, 1, 0], 15: [0, 0, 0, 1],  # immigration events
                16: [-1, 0, 1, 0],                     # x->z skipping y mutation (i.e. rate mu_base)
                17: [0, 0, 0, -1]}                     # special z2->z3 (z3 untracked) first-passage transitions

    def __str__(self):
        return str(self.params)

    def __iter__(self):
        return self.params

    def fbar(self, state):
        assert len(state) == self.numstates
        assert self.constant_growthandflowrates
        return sum([state[i]*self.growthrates[i] + self.flowrates[i] for i in xrange(self.numstates)]) / self.N

    def feedback_shape(self, param_name, state_coordinate, param_hill=PARAM_HILL, state_ratio=PARAM_Z0_RATIO):
        N = self.N
        p = state_coordinate

        if param_name == "alpha_plus":
            if self.feedback == "constant":
                print "Warning, feedback functionality in use for constant feedback setting (waste)"
                feedbackval = self.alpha_plus
            elif self.feedback == "hill":
                feedbackval = self.alpha_plus * (1 + p ** param_hill / (p ** param_hill + (state_ratio * N) ** param_hill))
            elif self.feedback == "step":
                feedbackval = None

        elif param_name == "alpha_minus":
            if self.feedback == "constant":
                print "Warning, feedback functionality in use for constant feedback setting (waste)"
                feedbackval = self.alpha_minus
            elif self.feedback == "hill":
                feedbackval = self.alpha_minus * (state_ratio*N)**param_hill / (p**param_hill + (state_ratio*N)**param_hill)
            elif self.feedback == "step":
                feedbackval = None

        elif param_name == "mu_base":
            if self.feedback == "constant":
                print "Warning, feedback functionality in use for constant feedback setting (waste)"
                feedbackval = self.mu_base
            elif self.feedback == "hill":
                feedbackval = self.mu_base * (1 + PARAM_GAMMA * p ** param_hill / (p ** param_hill + (state_ratio * N) ** param_hill))
            elif self.feedback == "step":
                feedbackval = None

        else:
            print "param_name %s not supported in feedback_shape" % param_name
            feedbackval = None

        return feedbackval

    def system_variants(self, init_cond, times):
        # TODO modified_params_list (the return) is different length for different self.system -- too messy, fix
        mod_params_dict = {}
        if self.numstates == 3:
            x, y, z = init_cond
            if self.system == "feedback_z":
                mod_params_dict['alpha_plus'] = self.feedback_shape("alpha_plus", z, state_ratio=PARAM_Z0_RATIO)
                mod_params_dict['alpha_minus'] = self.feedback_shape("alpha_minus", z, state_ratio=PARAM_Z0_RATIO)
                """
                alpha_plus = alpha_plus * (1 + z**PARAM_HILL / (z**PARAM_HILL + (PARAM_Z0_RATIO*N)**PARAM_HILL))
                alpha_minus = alpha_minus * (PARAM_Z0_RATIO*N)**PARAM_HILL / (z**PARAM_HILL + (PARAM_Z0_RATIO*N)**PARAM_HILL)
                """
            elif self.system == "feedback_yz":
                yz = y + z
                mod_params_dict['alpha_plus'] = self.feedback_shape("alpha_plus", yz, state_ratio=PARAM_Y0_PLUS_Z0_RATIO)
                mod_params_dict['alpha_minus'] = self.feedback_shape("alpha_minus", yz, state_ratio=PARAM_Y0_PLUS_Z0_RATIO)
                """
                alpha_plus = alpha_plus * (1 + yz**PARAM_HILL / (yz**PARAM_HILL + (PARAM_Y0_PLUS_Z0_RATIO*N)**PARAM_HILL))
                alpha_minus = alpha_minus * (PARAM_Y0_PLUS_Z0_RATIO*N)**PARAM_HILL / (yz**PARAM_HILL + (PARAM_Y0_PLUS_Z0_RATIO*N)**PARAM_HILL)
                """
        elif self.numstates == 2:
            if self.system == "feedback_mu_XZ_model":
                x, z = init_cond
                mod_params_dict['mu_base'] = self.feedback_shape("mu_base", z, state_ratio=PARAM_Z0_RATIO)
        else:
            if self.system == "feedback_XYZZprime":
                x, y, z, z2 = init_cond
                zsum = z + z2
                mod_params_dict['alpha_plus'] = self.feedback_shape("alpha_plus", zsum, state_ratio=PARAM_Z0_RATIO)
                mod_params_dict['alpha_minus'] = self.feedback_shape("alpha_minus", zsum, state_ratio=PARAM_Z0_RATIO)
                """
                alpha_plus = alpha_plus * (1 + zsum ** PARAM_HILL / (zsum ** PARAM_HILL + (PARAM_Z0_RATIO * N) ** PARAM_HILL))
                alpha_minus = alpha_minus * (PARAM_Z0_RATIO * N) ** PARAM_HILL / (zsum ** PARAM_HILL + (PARAM_Z0_RATIO * N) ** PARAM_HILL)
                """
        return mod_params_dict

    def ode_system_vector(self, init_cond, times):
        fbar = self.fbar(init_cond)
        mod_params_dict = self.system_variants(init_cond, times)
        p = self.mod_copy(mod_params_dict)  # TODO optimize, maybe slow, creates new params copy with augmented vals
        if self.numstates == 3:
            x, y, z = init_cond
            dxdt = p.v_x - x * (p.alpha_plus + p.mu_base) + y * p.alpha_minus + (p.a - fbar) * x
            dydt = p.v_y + x * p.alpha_plus - y * (p.alpha_minus + p.mu) + (p.b - fbar) * y
            dzdt = p.v_z + y * p.mu + x * p.mu_base + (p.c - fbar) * z
            return [dxdt, dydt, dzdt]
        elif self.numstates == 2:
            x, z = init_cond
            dxdt = p.v_x - x * p.mu_base + (p.a - fbar) * x
            dzdt = p.v_z + x * p.mu_base + (p.c - fbar) * z
            return [dxdt, dzdt]
        elif self.numstates == 4:
            x, y, z, z2 = init_cond
            dxdt = p.v_x - x * (p.alpha_plus + p.mu_base) + y * p.alpha_minus + (p.a - fbar) * x
            dydt = p.v_y + x * p.alpha_plus - y * (p.alpha_minus + p.mu) + (p.b - fbar) * y
            dzdt = p.v_z + y * p.mu + x * p.mu_base + (p.c - fbar) * z - z * p.mu
            dz2dt = p.v_z2 + z * p.mu + (p.c2 - fbar) * z2
            return [dxdt, dydt, dzdt, dz2dt]
        else:
            print "self.ode_system_vector not implemented for numstates >=5"
            return None

    def rxn_prop(self, state):
        """
        must correspond to self.update_dict (possibly store together as dictionary)
        note len(rxn_prop) should be len(self.update_dict.keys()) - 1 (since last key is for FPT conditional event)
        if flag_fpt, append rxn_prop with mu*last_state
        """
        fbar = self.fbar(state)  # TODO flag to switch N to x + y + z
        mod_params_dict = self.system_variants(state, None)
        p = self.mod_copy(mod_params_dict)  # TODO optimize, maybe slow, creates new params copy with augmented vals
        if self.numstates == 3:
            x_n, y_n, z_n = state
            rxn_prop = [p.a * x_n, fbar * (x_n),  # birth/death events for x  TODO: is it fbar*(x_n - 1)
                        p.b * y_n, fbar * (y_n),  # birth/death events for y  TODO: is it fbar*(y_n - 1)
                        p.c * z_n, fbar * (z_n),  # birth/death events for z  TODO: is it fbar*(z_n - 1)
                        p.alpha_plus * x_n, p.alpha_minus * y_n, p.mu * y_n,  # transition events
                        p.v_x, p.v_y, p.v_z,      # immigration events  #TODO maybe wrong
                        p.mu_base * x_n]          # special transition events (x->z)
        elif self.numstates == 2:
            x_n, z_n = state
            rxn_prop = [p.a * x_n, fbar * (x_n),    # birth/death events for x  TODO: is it fbar*(x_n - 1)
                        p.c * z_n, fbar * (z_n),    # birth/death events for z  TODO: is it fbar*(z_n - 1)
                        p.mu_base * x_n,            # transition events
                        p.v_x, p.v_z]               # immigration events  #TODO maybe wrong
        else:
            assert self.numstates == 4
            x_n, y_n, z_n, z2_n = state
            rxn_prop = [p.a * x_n, fbar * (x_n),     # birth/death events for x  TODO: is it fbar*(x_n - 1)
                        p.b * y_n, fbar * (y_n),     # birth/death events for y  TODO: is it fbar*(y_n - 1)
                        p.c * z_n, fbar * (z_n),     # birth/death events for z  TODO: is it fbar*(z_n - 1)
                        p.c2 * z2_n, fbar * (z2_n),  # birth/death events for z2  TODO: is it fbar*(z2_n - 1)
                        p.alpha_plus * x_n, p.alpha_minus * y_n,   # transition events xy, yx
                        p.mu * y_n, p.mu * z_n,                    # transition events yz, zz2
                        p.v_x, p.v_y, p.v_z, p.v_z2,                   # immigration events  #TODO maybe wrong
                        p.mu_base * x_n]                         # special transition events (x->z)
        return rxn_prop

    def get(self, param_label):
        # TODO implement (also modify params list attribute)
        return self.params_list[PARAMS_ID_INV[param_label]]  # could also use getattr

    def mod_copy(self, new_values):
        """
        new_values is dict of pairs of form param id: val
        return new params instance
        """
        #params_shift_list = self.params_list()
        params_dict_new = dict(self.params_dict)
        for k,v in new_values.iteritems():
            params_dict_new[k] = v
        return Params(params_dict_new, self.system, feedback=self.feedback)

    def printer(self):
        print "System: %s" % self.system
        print "Feedback: %s" % self.feedback
        for idx in xrange(len(PARAMS_ID.keys())):
            print "Param %d: (%s) = %.3f" % (idx, PARAMS_ID[idx], self.params[idx])

    def params_list(self):
        params_list = self.params_list[:]
        return params_list

    def write(self, filedir, filename):
        filepath = filedir + sep + filename
        with open(filepath, "wb") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for idx in xrange(len(PARAMS_ID.keys())):
                val = self.params_list[idx]
                if self.params_list[idx] is None:
                    val = 'None'
                writer.writerow([PARAMS_ID[idx], val])
            # any extra non-dynamics params
            writer.writerow(['system', self.system])
            writer.writerow(['feedback', self.feedback])
        return filepath

    @staticmethod
    def read(filedir, filename):
        with open(filedir + sep + filename, 'rb') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
            num_params = sum(1 for row in datareader)
            csvfile.seek(0)
            # prep dicts for reading
            params_dict = {k:None for k in PARAMS_ID_INV.keys()}
            aux_dict = {'system': None, 'feedback':None}
            if num_params != len(PARAMS_ID_INV.keys()) + len(aux_dict.keys()):
                print "Warning, unexpected params.csv line count"
            # iterate over contents
            for idx, pair in enumerate(datareader):
                if pair[0] in params_dict.keys():
                    params_dict[pair[0]] = pair[1]
                else:
                    aux_dict[pair[0]] = pair[1]
        assert aux_dict['system'] in ODE_SYSTEMS
        assert aux_dict['feedback'] in FEEDBACK_SHAPES
        return Params(params_dict, aux_dict['system'], init_cond=None, feedback=aux_dict['feedback'])