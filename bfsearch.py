#!/usr/bin/env python

from pykep_search.state_eph_grid import State, MOVE_TYPE
import sys


def bf_search(root):
    best = None
    states = [root.copy()]
    legs = 0
    while not states == []:
        frontier = []
        for s in states:
            for move in s.moves():
                ns = s.copy()
                if ns.next_move == MOVE_TYPE.TOF:
                    legs += 1
                ns.move(move)
                if not ns.isterminal():
                    frontier.append(ns)
                
                if legs % 100 == 0:
                    print '\r', legs,
                    sys.stdout.flush()

                    
                if ns.isfinal():
                    if best is None or ns.dv < best.dv:
                        best = ns.copy()
                        print '\r', best

        states = frontier
        
    print '\r', legs


if __name__ == '__main__':
    # bf_search(State())
    
    import cProfile
    cProfile.run('bf_search(State())')