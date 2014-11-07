#!/usr/bin/env python

import sys
import time
import random
import numpy as np
import functools

from pykep_search.state_eph_grid import State
from pykep_search.tools import pretty_time


UCT_C = 2
MAX_DV = 10000

class Node:
    def __init__(self, parent=None, state=None, last_move=None):
        self.Q = 0 # sum of values
        self.V = 0 # value estimate
        self.n = 0 # visits
        self.B = 0 # UCB1 bound
        self.untried_moves = list(state.moves())
        self.parent = parent
        self.children = []
        self.last_move = last_move


    def update(self, value, N=1, i=0):
        self.n += 1
        if self.parent is not None:
            #self.Q += value #1./self.n * (value - self.Q) # TODO: check this
            #self.B = self.Q/self.n + 0.7 * 0.01 * self.parent.n / self.n
            #self.B = self.Q/self.n + 2 * UCT_C * (N-i)/(1. * N) * np.sqrt(2*np.log(self.parent.n + 1)/self.n)
            self.V = max(self.V, value)
            #self.B = self.V + 0.7 * 0.01 * self.parent.n / self.n
            self.B = self.V + 2 * UCT_C * np.sqrt(2*np.log(self.parent.n+1)/self.n)

            
    def expand(self, state, move):
        n = Node(parent=self, state=state, last_move=move)
        self.untried_moves.remove(move)
        self.children.append(n)
        return n


    def select(self):
        # normal UCB1
        #return sorted(self.children, key=lambda n: n.B)[-1] # return child with highest value
        return self.children[np.argmax([c.B for c in self.children])]

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

        
def uct():
    best = None
    N = 1000000000
    n_rollouts = 0
    n_legs = 0
    start = time.time()

    rootstate = State()
    #rootstate.random_move()
    #rootstate.move('venus')
    root = Node(state=rootstate)

    max_select_depth = 0
    
    while n_rollouts < N+1:
        n_rollouts += 1
        state = rootstate.copy()
        node = root

        # select
        select_depth = 0
        while node.untried_moves == [] and node.children != []:
            node = node.select()
            state.move(node.last_move)
            select_depth += 1

        max_select_depth = max(max_select_depth, select_depth)

        # expand
        if node.untried_moves != [] and not state.isterminal():
            move = random.choice(node.untried_moves)
            state.move(move)
            node = node.expand(state, move)

        # check if leaf
        if not node == root and (state.isterminal() or node.children == [] and node.untried_moves == []):
            # detach child
            node.parent.children.remove(node)
            
        # rollout
        while not state.isterminal():
            state.random_move()

        # backpropagate
        value = 0
        if state.isfinal() and state.dv == state.dv: # TODO check why dv would be NaN?
            #print score, value            
            #value = max(MAX_DV - state.dv - 2000. * len(state.tof), 0.) / MAX_DV
            value = max(MAX_DV - state.dv, 0.)/ MAX_DV
            
        while node is not None:
            node.update(value) #, N=N, i=n_rollouts)
            node = node.parent


        n_legs += len(state.tof)
        
        if state.isfinal():
            if best is None or state.dv is not None and state.dv < best:
                best = state.dv
                print '\r%s' % str(state).ljust(80), 
                print
                sys.stdout.flush()


        if n_rollouts % 100 == 0:
            print '\r{0:,d} rollouts  {1:,d} legs  {2:.0f} legs/s  {3} {4} {5}'.format(n_rollouts, n_legs,
                                                                                   n_legs / (time.time() - start),
                                                                                   pretty_time(time.time() - start),
                                                                                   traverse(root),
                                                                                   max_select_depth
            ),
            #print '\r%d %d %d' % (n_rollouts, len(root.children), len(root.untried_moves)), 
            sys.stdout.flush()

#        if n_rollouts % 1000 == 0:
#            print
#            print ' '.join(['%s: %.4f' % (c.last_move, c.Q/c.n) for c in root.children[0].children])
#            sys.stdout.flush()
            
    print
    return root

if __name__=='__main__':
    root = uct()


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
