#!/usr/bin/env python

import sys
import time
import random
import numpy as np
import functools
import bisect
import math

from pykep_search.state_eph_grid import State, MOVE_TYPE, MAX_DV, fix_first_move, set_t_res
#from pykep_search.state_rosetta import State, MOVE_TYPE, MAX_DV, fix_first_move, set_t_res
from pykep_search.tools import pretty_time


UCT_C = .1
MAX_DV = 20000 #100000


def ucb1tuned(child):
    node = child.parent
    log_frac = math.log(node.n) / float(child.n)
    if child.n > 1:
        V = child.S / float(child.n-1) + math.sqrt(2 * log_frac)
    else:
        V = math.sqrt(2 * log_frac)
    return node.V + math.sqrt(log_frac * min(0.25, V))


class Node:
    def __init__(self, parent=None, state=None, last_move=None, c_P=0.0007):
        self.Q = 0 # sum of values
        self.V = 0 # value estimate
        self.n = 0 # visits
        self.B = 0 # UCB1 bound
        self.untried_moves = list(state.moves())
        self.parent = parent
        self.children = []
        self.last_move = last_move
        self.state = state.copy()
        self.c_P = c_P
        self.M = None # mean
        self.S = None # variance
        

    def update(self, value, N=1, i=0):
        self.n += 1
        self.V = max(self.V, value)
        if self.M is None:
            self.M = value
            self.S = 0
        else:
            last_M = self.M
            self.M = self.M + (value - self.M) / (1. * self.n)
            self.S = self.S + (value - last_M) * (value - self.M) # variance is sigma**2 = self.S / (self.n-1)
            
            
    def expand(self, state, move):
        n = Node(parent=self, state=state, last_move=move, c_P=self.c_P)
        self.untried_moves.remove(move)
        self.children.append(n)
        return n


    def select(self):
        self.children.sort(key=ucb1tuned) 
        return self.children[-1]
        
def traverse(node):
    if node.children == []:
        return 1
    else:
        return sum(map(traverse, node.children))
    

def collect(nodes, node):
    nodes.append(node)
    map(functools.partial(collect, nodes), node.children)

        
def uct(N=100000, c_P=0.0007, verbose=True):
    best = None
    best_n_legs = None
    n_rollouts = 0
    n_legs = 0
    start = time.time()

    rootstate = State()
    #rootstate.random_move()
    #rootstate.move('venus')
    root = Node(state=rootstate, c_P=c_P)

    max_select_depth = 0
    
    while n_rollouts < N+1:
        n_rollouts += 1
        node = root

        # select
        select_depth = 0
        while node.untried_moves == [] and node.children != []:
            node = node.select()
            #state.move(node.last_move) # this is expansive!
            #state = node.state

            select_depth += 1

        max_select_depth = max(max_select_depth, select_depth)

        # expand
        while node.untried_moves != []:
            move = random.choice(node.untried_moves)
            if node.state.next_move == MOVE_TYPE.TOF:
                n_legs += 1
            state = node.state.copy()
            state.move(move)
            node = node.expand(state, move)

#        # check if leaf
#        if state.isterminal() or node.children == [] and node.untried_moves == []:
#            if node == root:
#                # finish search
#                break
#                
#            # detach child
#            node.parent.children.remove(node)
            
#        # rollout
#        while not state.isterminal():
#            if state.next_move == MOVE_TYPE.TOF:
#                n_legs += 1
#            state.random_move()



            
        # backpropagate
        value = 0
        if node.state.isfinal() and node.state.dv == node.state.dv: # TODO check why dv would be NaN?
            value = max(MAX_DV - node.state.dv, 0.)/ MAX_DV
           
            if best is None or node.state.dv < best:
                best = node.state.dv
                best_n_legs = n_legs
                if verbose:
                    print '\r%s' % str(node.state).ljust(80), 
                    print
                    sys.stdout.flush()
            
        done = False
        while node is not None:
            node.update(value) #, N=N, i=n_rollouts)
            if node.children == [] and node.untried_moves == []:
                if node.parent is None:
                    done = True
                    break
                node.parent.children.remove(node)
            node = node.parent
        if done:
            break

        if verbose and n_rollouts % 100 == 0:
#            print '\r%d  ' % n_legs + ' '.join(['%s: %.4f %.4f' % (c.last_move[0], c.B, c.V) for c in root.children[-1].children]).ljust(100), 

            print '\r{0:,d} rollouts  {1:,d} legs  {2:.0f} legs/s  {3} {4} {5}'.format(n_rollouts, n_legs,
                                                                                   n_legs / (time.time() - start),
                                                                                   pretty_time(time.time() - start),
                                                                                   0, #traverse(root),
                                                                                   max_select_depth
            ).ljust(100),
            #print '\r%d %d %d' % (n_rollouts, len(root.children), len(root.untried_moves)), 
            sys.stdout.flush()

#        if n_rollouts % 1000 == 0:
#            print
#            print ' '.join(['%s: %.4f' % (c.last_move, c.Q/c.n) for c in root.children[0].children])
#            sys.stdout.flush()

        if n_legs >= N:
            break
            
    if verbose:
        print '\r{0:,d} rollouts  {1:,d} legs  {2:.0f} legs/s  {3}'.format(n_rollouts, n_legs,
                                                                           n_legs / (time.time() - start),
                                                                           pretty_time(time.time() - start),
                                                                           0, #traverse(root),
                                                                           max_select_depth
                                                                       ).ljust(100)
        sys.stdout.flush()
    return best, best_n_legs


if __name__=='__main__':
    # define problem
    set_t_res(32)
    fix_first_move(True)
    uct(N=50000, verbose=True)
