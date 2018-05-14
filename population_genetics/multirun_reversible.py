import matplotlib.pyplot as plt
import numpy as np

from simulate import popgen_simulate
from simulate_reversible import popgen_simulate_reversible


N_list = [1e2, 5*1e2, 1e3, 5*1e3, 1e4, 5*1e4, 1e5, 5*1e5, 1e6, 5*1e6, 1e7]

# parameters to replicate figure 5a of 2014 valley crossing paper
# for each param combo, they average over 500 independent runs
delta = -2e-4
s = 0.1
mu_0 = 1e-4
mu_0_backward = 1e-4
mu_1 = 1e-5
mutant_traits_norev = [(0.0, mu_0, 0),    # base pop (x)
                       (delta, mu_1, 0),  # 1-mutant (y)
                       (s, 0.0, 0)]       # 2-mutant (z)
mutant_traits_reversible = [(0.0, mu_0, 0.0),              # base pop (x)
                            (delta, mu_1, mu_0_backward),  # 1-mutant (y)
                            (s, 0.0, 0.0)]                 # 2-mutant (z)
mutant_traits_reversible_fast = [(0.0, mu_0, 0.0),                      # base pop (x)
                                 (delta, mu_1, 0.1*mu_0_backward),      # 1-mutant (y)
                                 (s, 0.0, 0.0)]                         # 2-mutant (z)


REPEATS=10
def get_average_run(N, mutant_traits, repeats=REPEATS):
    # TODO: should also be reporting stdev or variance
    trials = [0 for i in xrange(repeats)]
    for i in xrange(repeats):
        population, t = popgen_simulate_reversible(N, mutant_traits)
        trials[i] = t
    return sum(trials)/float(repeats), np.std(trials)


# simulate
t_norev = np.zeros(len(N_list))
t_reversible = np.zeros(len(N_list))
t_fast = np.zeros(len(N_list))
sd_norev = np.zeros(len(N_list))
sd_reversible = np.zeros(len(N_list))
sd_fast = np.zeros(len(N_list))

for i, N in enumerate(N_list):
    t_norev[i], sd_norev[i] = get_average_run(N, mutant_traits_norev)
    t_reversible[i], sd_reversible[i] = get_average_run(N, mutant_traits_reversible)
    t_fast[i], sd_fast[i] = get_average_run(N, mutant_traits_reversible_fast)

# plot output
plt.errorbar(N_list, t_norev, yerr=sd_norev, fmt='--s', label='not reversible')
plt.errorbar(N_list, t_reversible, yerr=sd_reversible, fmt='--o', label='reversible')
plt.errorbar(N_list, t_fast, yerr=sd_fast, fmt='--d', label='less reversible')
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
plt.xlabel('N')
plt.ylabel('time')
plt.legend(loc='upper right')
plt.title('Mean fixation time (avg %d runs per data point)' % REPEATS)

plt.show()
