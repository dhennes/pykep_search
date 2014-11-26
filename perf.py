#!/usr/bin/env python

import sys
import time
import random
import numpy as np
import functools
import bisect


import multiprocessing

#from pykep_search.state_eph_grid import State, MOVE_TYPE, MAX_DV, fix_first_move, set_t_res
from pykep_search.state_rosetta import State, MOVE_TYPE, MAX_DV, fix_first_move, set_t_res
from pykep_search.tools import pretty_time

import ucb1
from uct import uct
#from uct_ucb1tuned import uct # DANGER!
from random_search import random_search

def egreedy_run(i, N=20000):
    np.random.seed()
    return uct(N=N, c_P=0.04, verbose=False)

def ucb1_run(i, N=20000):
    np.random.seed()
    return uct(N=N, c_P=0.60, verbose=False) # this is just usb1tuned

def random_run(i, N=20000):
    np.random.seed()
    return random_search(N=N)
    

if __name__=='__main__':
    for n_legs in [100000]:
        print '#' * 80
        
        # define problem
        set_t_res(32)
        fix_first_move(False)

        from multiprocessing import Pool, Value
        from functools import partial

        N = 4000
        pool = Pool(4)
        res = []

        import pickle
        import os
        fname = 'results/perf/cassini/egreedy_full_%d.pkl' % n_legs
        dname = '/'.join(fname.split('/')[:-1])
        if not os.path.exists(dname):
            os.makedirs(dname)

        start = time.time()
        for i, r in enumerate(pool.imap_unordered(partial(egreedy_run, N=n_legs), xrange(N))):
            now = time.time()
            print '%d/%d  %.2f %d time: %s  remaining: %s' % (i+1, N, r[0], r[1],
                                                           pretty_time(now - start),
                                                           pretty_time((N-(i+1.))/(i+1.) * (now - start)))
            res.append(r)
        
            if (i+1) % 100 == 0:
                pickle.dump(res, open(fname, 'wb'))
