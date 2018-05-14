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
mutant_traits_fig5a = [(0.0, mu_0, 0),    # base pop
                       (delta, mu_1, 0),  # 1-mutant
                       (s, 0.0, 0)]       # 2-mutant
mutant_traits_reversible = [(0.0, mu_0, 0.0),              # base pop
                            (delta, mu_1, mu_0_backward),  # 1-mutant
                            (s, 0.0, 0.0)]                 # 2-mutant
mutant_traits_reversible_fast = [(0.0, mu_0, 0.0),                      # base pop
                                 (delta, mu_1, 0.1*mu_0_backward),      # 1-mutant
                                 (s, 0.0, 0.0)]                         # 2-mutant

REPEATS=10
def get_average_run(N, mutant_traits, repeats=REPEATS):
    trials = [0 for i in xrange(repeats)]
    for i in xrange(repeats):
        population, t = popgen_simulate_reversible(N, mutant_traits)
        trials[i] = t
    return sum(trials)/float(repeats)

# simulate
t_list_fig5a = [0 for N in N_list]
t_list_reversible = [0 for N in N_list]
t_list_fast = [0 for N in N_list]
for i, N in enumerate(N_list):
    t_fig5a = get_average_run(N, mutant_traits_fig5a)
    t_list_fig5a[i] = t_fig5a
    t_reversible = get_average_run(N, mutant_traits_reversible)
    t_list_reversible[i] = t_reversible
    t_fast = get_average_run(N, mutant_traits_reversible_fast)
    t_list_fast[i] = t_fast
    
# plot output
plt.plot(N_list, t_list_reversible, '--o', label='reversible')
plt.plot(N_list, t_list_fast, '--d', label='less reversible')
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
plt.xlabel('N')
plt.ylabel('time')
plt.legend(loc='upper right')
plt.title('Mean fixation time (avg %d runs per data point)' % REPEATS)


plt.show()
