#!/usr/bin/env python

import sys
import time

from pykep_search.state_eph_grid import State
from pykep_search.tools import pretty_time

if __name__=='__main__':
    best = None
    n_legs = 0
    n_rollouts = 0
    start = time.time()
    while True:
        s = State()
        while not s.isterminal() and not s.isfinal():
            s.random_move(continuous=False)

        n_rollouts += 1
        n_legs += len(s.tof)
            
        if s.isfinal():
            if best is None or s.dv is not None and s.dv < best:
                best = s.dv
                print '\r%s' % str(s).ljust(80), 
                print
                sys.stdout.flush()

        if n_rollouts % 100 == 0:
            print '\r{0:,d} rollouts  {1:,d} legs  {2:.0f} legs/s  {3}'.format(n_rollouts, n_legs,
                                                                               n_legs / (time.time() - start),
                                                                               pretty_time(time.time() - start)),
            sys.stdout.flush()
