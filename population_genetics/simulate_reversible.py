import numpy as np
import random

from constants import *


# NOTES
# note base fitness is 1
# k-mutant fitness is 1 + mutant_traits[k][0]
# form of mutant_traits is [... (w_k, mu_k_forward, mu_k_back) ...]
# population is dict of k:N_k mutant category to mutant pop

def count_pop(pop_dict):
    return sum(pop_dict.values())

def get_mean_fitness(pop_dict, mutant_traits, K, N):
    mutant_fitnesses = [1 + pair[0] for pair in mutant_traits]
    mean_fitness = sum(mutant_fitnesses[i]*pop_dict[i] for i in xrange(K+1)) / N
    return mean_fitness

def sample_events(pop_dict, k, dt, mutant_traits, mean_fitness):
    mutant_fitness = mutant_traits[k][0] + 1
    p_a = dt*(1 + mutant_fitness - mean_fitness)     # divide
    p_b1 = dt*mutant_traits[k][1]                    # mutate forward
    p_b2 = dt*mutant_traits[k][2]                    # mutate back
    p_c = dt                                         # die
    n = pop_dict[k]
    divisions = np.random.binomial(n, p_a)
    mutations_forward = np.random.binomial(n, p_b1)
    mutations_backward = np.random.binomial(n, p_b2)
    deaths = np.random.binomial(n, p_c)
    return divisions, mutations_forward, mutations_backward, deaths
    
def increment_pop(increment_list, pop_dict, K, N):
    for k in xrange(K+1):
        pop_dict[k] += increment_list[k]
        if pop_dict[k] < 0:
            print pop_dict[k], "(setting to zero)"
            pop_dict[k] = 0
    if count_pop(pop_dict) != N:
        return normalize_pop(pop_dict, K, N)
    else:
        return pop_dict

def normalize_pop(pop_dict, K, N):
    weight = float(N) / count_pop(pop_dict)
    for k in xrange(K+1):
        pop_dict[k] = int(round(pop_dict[k]*weight))
    return pop_dict

def popgen_simulate_reversible(N=DEFAULT_N, mutant_traits=DEFAULT_MUTANT_TRAITS_REVERSIBLE, dt=DEFAULT_DT):

    # setup
    K = len(mutant_traits) - 1
    population = {k:0 for k in xrange(K+1)}
    population[0] = N  # note initial pop assumed to have 0 mutants
    print " simulating a %d-hit process on %d individuals" % (K, N)

    # ensure mutation fitness deviations are well-defined  
    #assert(1.0/float(N) < mutant_traits[-1][0] < 1.0)  # K-mutant moderately fit
    assert(mutant_traits[-1][1] == 0.0)  # K-mutants don't mutate forward
    assert(mutant_traits[-1][2] >= 0.0)  # K-mutants may mutate backward
    assert(mutant_traits[0][2] == 0.0)  # 0-mutants don't mutate backward
    for i in xrange(K):
        assert(mutant_traits[i][0] <= 0)  # intermediates less fit
        assert(mutant_traits[i][1] > 0)   # intermediates must mutate forward
        assert(mutant_traits[i][2] >= 0)  # intermediates may mutate backward

    # SIMULATE (see p9, Section 5 of paper)
    # 0. set timestep dt (they use dt = 10^-2 = 0.01 generations)
    # 1. calculate mean fitness w_bar
    # 2. each k-mutant does either A, B1 or B2, or C independently
    #        A: divides into two k_mutants, prob (1 + w_k - w_bar)*dt
    #        B1: divides with one k-mutant, one k+1 mutant, prob (mu_k_forward)*dt
    #        B2: divides with one k-mutant, one k-1 mutant, prob (mu_k_back)*dt
    #        C: dies, prob (1)*dt
    #        note there are thus 3N indep events per timestep
    # 3. if pop N* after 2. is not N, then multiply each N_k by N/N* and round to nearest int
    # 4. repeat until N_K = N (i.e. whole pop is K-mutants)
    # 5. at end of run, return last time there were no K-mutants, call this
    #    "time to production of first successful K-mutant"
    t = 0.0
    while 1:
        t += dt
        w_bar = get_mean_fitness(population, mutant_traits, K, N)
        increments = [0 for k in xrange(K+1)]
        for k in xrange(K+1):
            divisions, mutations_forward, mutations_backward, deaths = sample_events(population, k, dt, mutant_traits, w_bar)
            increments[k] += divisions - mutations_forward - mutations_backward - deaths
            if mutations_forward != 0:
                increments[k+1] += mutations_forward
            if mutations_backward != 0:
                increments[k-1] += mutations_backward
        population = increment_pop(increments, population, K, N)
        #print population
        if population[K] == N:
            print "K-mutants have fixated at time %.2f (%d steps)" % (t, t/dt)
            break
    return population, t

if __name__ == '__main__':
    popgen_simulate_reversible()
