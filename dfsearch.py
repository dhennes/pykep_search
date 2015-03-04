#!/usr/bin/env python

import sys
import numpy as np

#from pykep_search.state_eph_grid import set_t_res, fix_first_move, State, MOVE_TYPE
from pykep_search.state_rosetta import set_t_res, fix_first_move, State, MOVE_TYPE


LEGS = 0
BEST = None


def df_search(state):
    global BEST
    global LEGS
    
    if state.isfinal():
        if BEST is None or state.dv < BEST.dv:
            BEST = state.copy()
            print '\r', BEST
            return

    if state.isterminal():
        return

    moves = np.random.permutation(state.moves())
    for m in moves:
        ns = state.copy()
        if ns.next_move == MOVE_TYPE.TOF:
            LEGS += 1

        if LEGS % 100 == 0:
            print '\r', '{:,d}'.format(LEGS),
            sys.stdout.flush()
            
        ns.move(m)
        df_search(ns)


if __name__ == '__main__':
    X = [2**t for t in range(2, 6)]
    Y = []
    F = []
    for x in X:
        BEST = None
        set_t_res(x)
        fix_first_move(True)
        state = State()    
        df_search(state)
        print '\r> ', x, LEGS
        Y.append(LEGS)
        F.append(BEST.dv)
        print 'X = %s\nY = %s\nF = %s' % (str(X), str(Y), str(F))
                
    import matplotlib.pylab as plt
    plt.plot(X, Y)
    plt.xlabel('grid resolution')
    plt.ylabel('log(legs)')
    plt.gca().set_yscale('log')
    plt.show()
