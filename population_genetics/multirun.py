import matplotlib.pyplot as plt
import numpy as np

from simulate import popgen_simulate

# parameters to replicate figure 5a of 2014 valley crossing paper
# for each param combo, they average over 500 independent runs
delta = -2e-4
s = 0.1
mu_0 = 1e-5
mu_1 = 1e-4
mutant_traits = [(0.0, mu_0),    # base pop
                 (delta, mu_1),  # 1-mutant
                 (s, 0.0)]       # 2-mutant
N_list = [1e2, 5*1e2, 1e3, 5*1e3, 1e4, 5*1e4, 1e5, 5*1e5, 1e6, 5*1e6, 1e7]


def get_average_run(N, mutant_traits, repeats=10):
    trials = [0 for i in xrange(repeats)]
    for i in xrange(repeats):
        population, t = popgen_simulate(N, mutant_traits)
        trials[i] = t
    return sum(trials) / float(repeats)


t_list = [0 for N in N_list]
for i, N in enumerate(N_list):
    population, t = get_average_run(N, mutant_traits)
    t_list[i] = t

# plot output
plt.plot(N_list, t_list, 'x')
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
plt.xlabel('N')
plt.ylabel('time')
plt.show()
