#!/usr/bin/env python

import sys
import time
import random
import numpy as np
import functools
import bisect

from pykep_search.state_eph_grid import State, MOVE_TYPE, MAX_DV, fix_first_move, set_t_res
from pykep_search.tools import pretty_time


UCT_C = .19
MAX_DV = 20000 #100000

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
        

    def update(self, value, N=1, i=0):
        self.n += 1
        if self.parent is not None:
            #self.Q += value #1./self.n * (value - self.Q) # TODO: check this
            #self.B = self.Q/self.n + 0.7 * 0.01 * self.parent.n / self.n
            #self.B = self.Q/self.n + 2 * UCT_C * (N-i)/(1. * N) * np.sqrt(2*np.log(self.parent.n + 1)/self.n)

            #self.Q += value
            #self.V = self.Q * 1. / self.n 
            
            self.V = max(self.V, value)

            #self.B = self.V + 0.7 * 0.01 * self.parent.n / self.n
            #self.B = self.V + 2 * UCT_C * np.sqrt(2*np.log(self.parent.n+1)/self.n)
    
        #if self.parent is not None:
        #    self.parent.children.sort()
            
    def expand(self, state, move):
        n = Node(parent=self, state=state, last_move=move, c_P=self.c_P)
        self.untried_moves.remove(move)
        self.children.append(n)
        #bisect.insort(self.children, n)
        return n


    def select(self):
        # normal UCB1
        #return sorted(self.children)[-1]
        #return self.children[np.argmax([c.B for c in self.children])]

        #for c in self.children:
        #    c.B = c.V + 0.7 * 0.01 * self.n/c.n
        #    #c.B = c.V + 2 * UCT_C * np.sqrt(2*np.log(self.n/c.n))

        self.children.sort(key=lambda c: c.V + self.c_P * np.sqrt(np.log(self.n/c.n))) 
        #self.children.sort(key=lambda c: c.V + self.c_P * self.n/c.n)
        return self.children[-1]

        #return random.choice(self.children)
        
        # # boltzmann selection
        #e = np.exp([c.Q/c.n for c in self.children])
        #return self.children[np.searchsorted(np.cumsum(e/sum(e)), np.random.uniform())]

        

        
        

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


def uct_run(i, N=20000):
    return uct(N=N, verbose=False)

    
if __name__=='__main__':
    # define problem
    set_t_res(36)
    fix_first_move(True)

    from multiprocessing import Pool
    from functools import partial
    
    pool = Pool(32)
    res = pool.map(partial(uct_run, N=20000), xrange(100))

    import matplotlib.pylab as plt
    plt.scatter(xrange(1, len(res)+1), sorted(res))
    plt.show()

    # import cProfile
    # cProfile.run('uct()')


    # import networkx as nx
    # import matplotlib.pylab as plt

    # try:
    #     from networkx import graphviz_layout
    # except ImportError:
    #         raise ImportError("This example needs Graphviz and either PyGraphviz or Pydot")

    # G = nx.Graph()
    # nodes = []
    # collect(nodes, root)
    # print len(nodes)
    # G.add_nodes_from(nodes)
    # for i, n in enumerate(nodes):
    #     G.add_edges_from([(i, nodes.index(c)) for c in n.children])


    # pos=nx.graphviz_layout(G,prog='twopi',args='')
    # nx.draw(G,pos,node_size=20,alpha=0.5,node_color="blue", with_labels=False)
    # plt.show()
