#!/usr/bin/env python

import sys
import time

#from pykep_search.state_eph_grid import State
#from pykep_search.state_eph_grid import State, MOVE_TYPE, MAX_DV, fix_first_move, set_t_res

from pykep_search.state_cassini import State, MOVE_TYPE, MAX_DV, fix_first_move, set_t_res
#from pykep_search.state_rosetta import State, MOVE_TYPE, MAX_DV, fix_first_move, set_t_res
from pykep_search.tools import pretty_time

def random_search(N=10000, verbose=False):
    best = None
    n_legs = 0
    n_rollouts = 0
    start = time.time()
    while True:
        s = State()
        while not s.isterminal() and not s.moves() == []:
            if s.next_move == MOVE_TYPE.TOF:
                n_legs += 1
            s.random_move(continuous=False)
            if n_legs >= N:
                break

        n_rollouts += 1

        if s.isfinal():
            if best is None or s.dv is not None and s.dv < best:
                best = s.dv
                if verbose:
                    print '\r%s' % str(s).ljust(80), 
                    print
                    sys.stdout.flush()
                    
        if verbose and n_rollouts % 100 == 0:
            print '\r{0:,d} rollouts  {1:,d} legs  {2:.0f} legs/s  {3}'.format(n_rollouts, n_legs,
                                                                               n_legs / (time.time() - start),
                                                                               pretty_time(time.time() - start)),
            sys.stdout.flush()

        if n_legs >= N:
            break

    if verbose:
        print '\r{0:,d} rollouts  {1:,d} legs  {2:.0f} legs/s  {3}'.format(n_rollouts, n_legs,
                                                                               n_legs / (time.time() - start),
                                                                               pretty_time(time.time() - start)),
        sys.stdout.flush()
    return best


if __name__=='__main__':
    # define problem
    set_t_res(32)
    fix_first_move(False)
    random_search(N=50000, verbose=True)

