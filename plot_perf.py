#!/usr/bin/env python

import pickle
import numpy as np
import matplotlib.pylab as plt


NUM_TRIALS = 10000

def ecdf(fname, N = 100000,  N_budget = 1e7, dv_target=6760.012894307366+50.):
    print fname
    data = pickle.load(open(fname, 'rb'))
    dv, ns = zip(*data)
    dv = np.array(dv)
    ns = np.array(ns)

    print 'N_runs:', len(ns)

    converged = np.where(dv <= dv_target)[0]
    p_s = len(converged) * 1. / len(dv)
    print 'p_s: ', p_s

    if p_s == 0.:
        return None, None
    
    RT_s = ns[converged].mean()
    print 'RT_s: ', RT_s

    E_RT = RT_s + (1-p_s)/p_s * N
    print 'E[RT]: ', E_RT

    # compile trials
    trials = 0
    succ_trials = []
    failed = 0
    runs = (d for d in data)

    while trials < NUM_TRIALS:
        # randomly sample
        d, n = data[np.random.randint(0, len(data))]

        # try:
        #     d, n = runs.next()
        # except StopIteration:
        #     break

        if d <= dv_target:
            # sucess
            trials += 1
            succ_trials.append(n + failed * N)
            failed = 0
            continue

        failed += 1

        if failed * N >= N_budget:
            trials += 1 
            failed = 0       

    print 'succ. trials: ', len(succ_trials), trials

    succ_trials = sorted(succ_trials)
    perc = np.array(range(len(succ_trials))) + 1.0
    perc = perc / trials * 1.
    perc = perc.tolist()

    #print succ_trials, perc

    return succ_trials, perc


if __name__ == '__main__':
    N=50000
    N_budget=1e8
    dv_target = 6760.012894307366 # cassini fixed t0
    dv_target += 50.

    fname = 'results/perf/cassini/egreedy_%d.pkl' % N
    succ_trials, perc = ecdf(fname, N=N, N_budget=N_budget, dv_target=dv_target)
    succ_trials.insert(0, succ_trials[0])
    perc.insert(0, 0)
    succ_trials.append(N_budget)
    perc.append(perc[-1])
    plt.step(succ_trials, perc, where='post', color='k')

    # --------

    fname = 'results/perf/cassini/ucb1_%d.pkl' % N
    succ_trials, perc = ecdf(fname, N=N, N_budget=N_budget, dv_target=dv_target)
    succ_trials.insert(0, succ_trials[0])
    perc.insert(0, 0)
    succ_trials.append(N_budget)
    perc.append(perc[-1])
    plt.step(succ_trials, perc, where='post', color='k', linestyle=':')

    # --------

#     fname = 'results/perf/cassini/ucb1tuned_%d.pkl' % N
#     succ_trials, perc = ecdf(fname, N=N, N_budget=N_budget, dv_target=dv_target)
#     succ_trials.insert(0, succ_trials[0])
#     perc.insert(0, 0)
#     succ_trials.append(N_budget)
#     perc.append(perc[-1])
#     plt.step(succ_trials, perc, where='post', color='k', linestyle='--')

    # ********

    ax = plt.gca()
    plt.ylim(0., 1.)
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xlabel('Lambert legs', fontsize=16)
    ax.set_ylabel('proportion of successful trails', fontsize=16)

    plt.legend(['$\epsilon$-greedy', 'UCB1'], fontsize=16, loc='upper left')
    plt.savefig('paper/figures/ecdf_cassini_50K.eps', bbox_inches='tight')


    # ==============
    plt.clf()

    N=50000
    N_budget=1e8
    dv_target=6498.0430789025195
    dv_target += 50.

    fname = 'results/perf/rosetta/egreedy_%d.pkl' % N
    succ_trials, perc = ecdf(fname, N=N, N_budget=N_budget, dv_target=dv_target)
    succ_trials.insert(0, succ_trials[0])
    perc.insert(0, 0)
    succ_trials.append(N_budget)
    perc.append(perc[-1])
    plt.step(succ_trials, perc, where='post', color='k')

    # --------

    fname = 'results/perf/rosetta/ucb1_%d.pkl' % N
    succ_trials, perc = ecdf(fname, N=N, N_budget=N_budget, dv_target=dv_target)
    succ_trials.insert(0, succ_trials[0])
    perc.insert(0, 0)
    succ_trials.append(N_budget)
    perc.append(perc[-1])
    plt.step(succ_trials, perc, where='post', color='k', linestyle=':')

    # --------

    fname = 'results/perf/rosetta/ucb1tuned_%d.pkl' % N
    succ_trials, perc = ecdf(fname, N=N, N_budget=N_budget, dv_target=dv_target)
    succ_trials.insert(0, succ_trials[0])
    perc.insert(0, 0)
    succ_trials.append(N_budget)
    perc.append(perc[-1])
    plt.step(succ_trials, perc, where='post', color='k', linestyle='--')

    # --------
    
    ax = plt.gca()
    plt.ylim(0., 1.)
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xlabel('Lambert legs', fontsize=16)
    ax.set_ylabel('proportion of successful trails', fontsize=16)

    plt.legend(['$\epsilon$-greedy', 'UCB1', 'UCB1-Tuned'], fontsize=16, loc='upper left')
    plt.savefig('paper/figures/ecdf_rosetta_50K.eps', bbox_inches='tight')

    # ==============
    plt.clf()

    N=100000
    N_budget=1e9
    dv_target=6571.4863484566258
    dv_target += 50.

    fname = 'results/perf/cassini/egreedy_full_%d_cassini.pkl' % N
    succ_trials, perc = ecdf(fname, N=N, N_budget=N_budget, dv_target=dv_target)
    succ_trials.insert(0, succ_trials[0])
    perc.insert(0, 0)
    succ_trials.append(N_budget)
    perc.append(perc[-1])
    plt.step(succ_trials, perc, where='post', color='k')

#     # --------

#     fname = 'results/perf/cassini/ucb1_full_%d.pkl' % N
#     succ_trials, perc = ecdf(fname, N=N, N_budget=N_budget, dv_target=dv_target)
#     succ_trials.insert(0, succ_trials[0])
#     perc.insert(0, 0)
#     succ_trials.append(N_budget)
#     perc.append(perc[-1])
#     plt.step(succ_trials, perc, where='post', color='k', linestyle=':')

#     fname = 'results/perf/cassini/ucb1tuned_full_%d.pkl' % N
#     succ_trials, perc = ecdf(fname, N=N, N_budget=N_budget, dv_target=dv_target)
#     succ_trials.insert(0, succ_trials[0])
#     perc.insert(0, 0)
#     succ_trials.append(N_budget)
#     perc.append(perc[-1])
#     plt.step(succ_trials, perc, where='post', color='k')

    # --------
    
    ax = plt.gca()
    plt.ylim(0., 1.)
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xlabel('Lambert legs', fontsize=16)
    ax.set_ylabel('proportion of successful trails', fontsize=16)

    plt.legend(['$\epsilon$-greedy', 'UCB1'], fontsize=16, loc='upper left')
    plt.savefig('paper/figures/ecdf_cassini_full_100K.eps', bbox_inches='tight')

    # ==============
    plt.clf()

    N=100000
    N_budget=1e9
    dv_target=5887.
    dv_target += 5.

    fname = 'results/perf/cassini/egreedy_full_%d.pkl' % N # this is correct (rosseta)
    succ_trials, perc = ecdf(fname, N=N, N_budget=N_budget, dv_target=dv_target)
    succ_trials.insert(0, succ_trials[0])
    perc.insert(0, 0)
    succ_trials.append(N_budget)
    perc.append(perc[-1])
    plt.step(succ_trials, perc, where='post', color='k')

    # --------

    fname = 'results/perf/cassini/ucb1_full_%d.pkl' % N
    succ_trials, perc = ecdf(fname, N=N, N_budget=N_budget, dv_target=dv_target)
    succ_trials.insert(0, succ_trials[0])
    perc.insert(0, 0)
    succ_trials.append(N_budget)
    perc.append(perc[-1])
    plt.step(succ_trials, perc, where='post', color='k', linestyle=':')

#     fname = 'results/perf/cassini/ucb1tuned_full_%d.pkl' % N
#     succ_trials, perc = ecdf(fname, N=N, N_budget=N_budget, dv_target=dv_target)
#     succ_trials.insert(0, succ_trials[0])
#     perc.insert(0, 0)
#     succ_trials.append(N_budget)
#     perc.append(perc[-1])
#     plt.step(succ_trials, perc, where='post', color='k')

    # --------
    
    ax = plt.gca()
    plt.ylim(0., 1.)
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xlabel('Lambert legs', fontsize=16)
    ax.set_ylabel('proportion of successful trails', fontsize=16)

    plt.legend(['$\epsilon$-greedy', 'UCB1'], fontsize=16, loc='upper left')
    plt.savefig('paper/figures/ecdf_rosetta_full_100K.eps', bbox_inches='tight')



