import random

# DEFINITIONS
# note base fitness is 1
# k-mutant fitness is 1 + mutant_traits[k][0]
# form of mutant_traits is [... (w_k, mu_k) ...]
# population is dict of k:N_k mutant categry to mutant pop

N = 100  # total fixed pop
mutant_traits = [(0.0, 0.1),   # base pop
                 (-0.1, 0.1),  # 1-mutant
                 (-0.2, 0.1),  # 2-mutant etc
                 (0.05, 0.0)]
mutant_fitness = [1 + pair[0] for pair in mutant_traits]
K = len(mutant_traits) - 1
print " simulating a %d-hit process on %d individuals" % (K, N)

# ensure mutation fitness deviations are well-defined  
assert(1.0/float(N) < mutant_traits[-1][0] < 1.0)  # K-mutant moderately fit
assert(mutant_traits[-1][1] == 0.0)  # K-mutants don't mutate
for i in xrange(K):
    assert(mutant_traits[i][0]) <= 0  # intermediates less fit
    assert(mutant_traits[i][1]) > 0  # intermediates can mutate 


def count_pop(pop_dict):
    return sum(pop_dict.values())

# INITIALIZE
population[0] = N  # note initial pop assumed to have 0 mutants


# SIMULATE
# 0. set timestep dt (they use dt = 10^-2 = 0.01 generations)
# 1. calculate mean fitness w_bar
# 2. each k-mutant does either A, B, or C independently
#        A: divides into two k_mutants, prob (1 + w_k - w_bar)*dt
#        B: divides with one k-mutant, one k+1 mutant, prob (mu_k)*dt
#        C: dies, prob (1)*dt
#        note there are thus 3N indep events per timestep
# 3. if pop N* after 2. is not N, then multiply each N_k by N/N* and round to nearest int
# 4. repeat until N_K = N (i.e. whole pop is K-mutants)
# 5. at end of run, return last time there were no K-mutants, call this
#    "time to production of first successful K-mutant"

#number_to_mutate
#random.sample(range[N], number_to_mutate)
