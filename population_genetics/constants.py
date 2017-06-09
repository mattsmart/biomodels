DEFAULT_N = 10000  # total fixed pop
DEFAULT_MUTANT_TRAITS = [(0.0, 0.1),   # base pop
                         (-0.1, 0.1),  # 1-mutant
                         (-0.2, 0.1),  # 2-mutant etc
                         (0.05, 0.0)]
DEFAULT_MUTANT_TRAITS_REVERSIBLE = [(0.0, 0.1, 0.0),   # base pop (fitness, forward rate, back rate)
                                    (-0.1, 0.1, 0.2),  # unstable
                                    (0.5, 0.0, 0.0)]  # mutant
DEFAULT_DT = 0.01  # simulation stepsize
