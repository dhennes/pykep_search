#!/usr/bin/env python

import sys
import numpy as np

from pykep_search.state_eph_grid import State, MOVE_TYPE


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
            print '\r', LEGS,
            sys.stdout.flush()
            
        ns.move(m)
        df_search(ns)


if __name__ == '__main__':
    state = State()
    df_search(state)
    print '\r', LEGS