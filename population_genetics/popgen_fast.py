import numpy as np
import random

# DEFINITIONS
# note base fitness is 1
# k-mutant fitness is 1 + mutant_traits[k][0]
# form of mutant_traits is [... (w_k, mu_k) ...]
# population is dict of k:N_k mutant categry to mutant pop

N = 10000  # total fixed pop
mutant_traits = [(0.0, 0.1),   # base pop
                 (-0.1, 0.1),  # 1-mutant
                 (-0.2, 0.1),  # 2-mutant etc
                 (0.05, 0.0)]
mutant_fitness = [1 + pair[0] for pair in mutant_traits]
K = len(mutant_traits) - 1
population = {k:0 for k in xrange(K+1)}
print " simulating a %d-hit process on %d individuals" % (K, N)

# ensure mutation fitness deviations are well-defined  
assert(1.0/float(N) < mutant_traits[-1][0] < 1.0)  # K-mutant moderately fit
assert(mutant_traits[-1][1] == 0.0)  # K-mutants don't mutate
for i in xrange(K):
    assert(mutant_traits[i][0] <= 0)  # intermediates less fit
    assert(mutant_traits[i][1] > 0)  # intermediates can mutate 


# FUNCTIONS

def count_pop(pop_dict):
    return sum(pop_dict.values())

def get_mean_fitness(pop_dict):
    mean_fitness = sum(mutant_fitness[i]*pop_dict[i] for i in xrange(K+1)) / N
    return mean_fitness

def sample_events(pop_dict, k, dt, mean_fitness):
    p_a = dt*(1 + mutant_fitness[k] - mean_fitness)  # divide
    p_b = dt*mutant_traits[k][1]                     # mutate
    p_c = dt                                         # die
    n = pop_dict[k]
    divisions = np.random.binomial(n, p_a)
    mutations = np.random.binomial(n, p_b)
    deaths = np.random.binomial(n, p_c)
    return divisions, mutations, deaths
    
def increment_pop(increment_list, pop_dict):
    for k in xrange(K+1):
        pop_dict[k] += increment_list[k]
    if count_pop(pop_dict) != N:
        return normalize_pop(pop_dict)
    else:
        return pop_dict

def normalize_pop(pop_dict):
    weight = float(N) / count_pop(pop_dict)
    for k in xrange(K+1):
        pop_dict[k] = int(round(pop_dict[k]*weight))
    return pop_dict


# INITIALIZE
population[0] = N  # note initial pop assumed to have 0 mutants
print get_mean_fitness(population)
print count_pop(population)

# SIMULATE (see p9, Section 5 of paper)
dt = 0.01
t = 0.0
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
while 1:
    t += dt
    w_bar = get_mean_fitness(population)
    increments = [0 for k in xrange(K+1)]
    for k in xrange(K+1):
        divisions, mutations, deaths = sample_events(population, k, dt, w_bar)
        increments[k] += divisions - mutations - deaths
        if mutations != 0:
            increments[k+1] += mutations
    population = increment_pop(increments, population)
    print population
    if population[K] == N:
        print "K-mutants have fixated at time %.2f (%d steps)" % (t, t/dt)
        break

print "broke from loop"
    
                
    
#number_to_mutate
#random.sample(range[N], number_to_mutate)
