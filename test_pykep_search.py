#!/usr/bin/env python

"""Performance tests for pykep_search"""

import sys
import types
import time
import numpy as np

import PyKEP as kep

from pykep_search.tools import lambert_leg
from pykep_search.state import State


def timeit(func):
    """Time a function."""
    start = time.time()
    func()
    end = time.time()
    print '{0} {1:>{2}.2f} secs'.format(func.__name__, end - start, 70-len(func.__name__))


def test_lambert_leg_launch(n=10000):
    for i in xrange(n):
        t0 = np.random.uniform(-71048., 16263.)
        tof = np.random.uniform(0., 2000.)
        lambert_leg('earth', 'venus', t0, tof)


def test_lambert_leg_flyby(n=10000):
    for i in xrange(n):
        t0 = np.random.uniform(-71048., 16263.)
        tof = np.random.uniform(0., 2000.)
        vrel = np.random.rand(3).tolist()
        lambert_leg('earth', 'venus', t0, tof, vrel=vrel)


def test_state_random_move(n=1000):
    for i in xrange(n):
        s = State()
        while not s.isterminal():
            s.random_move()


def test_state_random_move_continuous(n=1000):
    for i in xrange(n):
        s = State()
        while not s.isterminal():
            s.random_move(continuous=True)


def test_state_copy():
    a = State()
    a.seq.append('foo')
    a.tof.append(1.23)
    a.dv = 1.23
    b = a.copy()
    c = State()
    assert a.seq == b.seq and a.tof == b.tof and a.dv == b.dv, 'copy() error: %s is not %s'
    assert not c.seq == a.seq, 'reference error'
    
            
            
if __name__=='__main__':
    mod = sys.modules[__name__]
    map(timeit, [mod.__dict__.get(name) for name in sorted(dir(mod))
                 if isinstance(mod.__dict__.get(name), types.FunctionType)
                 and name.startswith('test_')])