import copy
import numpy as np

import tools


MAX_EPOCH = 15000.

BRANCH_FACTOR = 10

MOVE_TYPE = tools.enum('T0', 'PLANET', 'TOF')
MOVES = {
#    MOVE_TYPE.T0: np.linspace(6456.0, 8456.0, BRANCH_FACTOR),
#    MOVE_TYPE.T0: np.linspace(-1000.0, 0.0, BRANCH_FACTOR),
    MOVE_TYPE.T0: np.linspace(-775., -805, BRANCH_FACTOR),
#    MOVE_TYPE.PLANET: ['earth', 'venus', 'mars', 'jupiter'], #, 'saturn'],
    MOVE_TYPE.PLANET: ['earth', 'venus', 'mars', 'jupiter', 'saturn'],
    MOVE_TYPE.TOF: np.linspace(30.0, 5000.0, BRANCH_FACTOR),
}


class State():
    """State class."""
    
    def __init__(self, seq=['earth'], t0=None, tof=[], vrel=None, dv=0, next_move=MOVE_TYPE.T0):
        self.seq = copy.copy(seq)
        self.tof = copy.copy(tof)
        self.t0 = t0
        self.vrel = copy.copy(vrel)
        self.dv = dv
        self.next_move = next_move
        

    def moves(self):
        return [] if self.isterminal() else MOVES[self.next_move]


    def move(self, move):
#        assert move in self.moves(), '%s is not in current move list' % move
#        assert not self.isterminal(), 'current state %s is terminal' % self
#        assert not self.final(), 'current state %s is final' % self
        if self.next_move == MOVE_TYPE.T0:
            self.t0 = move
            self.next_move = MOVE_TYPE.PLANET
            return
        elif self.next_move == MOVE_TYPE.PLANET:
            self.seq.append(move)
            self.next_move = MOVE_TYPE.TOF
            return
        elif self.next_move == MOVE_TYPE.TOF:
            self.tof.append(move)
            self.next_move = MOVE_TYPE.PLANET
            # evaluate lambert leg
            t = self.t0 + sum(self.tof[:-1])
            tof = self.tof[-1]
            dv, vrel_out = tools.lambert_leg(self.seq[-2], self.seq[-1], t, tof,
                                             vrel=self.vrel)
                                             #, rendezvous=s.isfinal())

            self.dv += dv
            self.vrel = vrel_out
            return
        else:
            print 'unknown move type %s' % self.next_move


    def random_move(self, continuous=False):
        # continuous random move
        if continuous and (self.next_move == MOVE_TYPE.T0 or self.next_move == MOVE_TYPE.TOF):
            move = np.random.uniform(min(MOVES[self.next_move]), max(MOVES[self.next_move]))
        else:
            moves = self.moves()
            move = moves[np.random.randint(0, len(moves))]
        self.move(move)
        return move
            
            
    def isterminal(self):
        if self.dv is not None and self.dv > 10000:
            return True
        if self.t0 is not None and self.t0 + sum(self.tof) > MAX_EPOCH:
            return True
        if self.isfinal():
            return True
        return False


    def isfinal(self):
        return self.seq[-1] == MOVES[MOVE_TYPE.PLANET][-1] and len(self.seq) -1 == len(self.tof)
        

    def get_value():
        return self.dv

        
    def copy(self):
        return copy.deepcopy(self)
      

    def __key(self):
        return (self.seq, self.t0, self.tof)


    def __eq__(s1, s2):
        return s1.__key() == s2.__key()


    def __hash__(self):
        return hash(self.__key())
        
        
    def __repr__(self):
        s = '{:8.2f} m/s  '.format(self.dv)
        s += '{:7.2f} days  '.format(sum(self.tof))
        s += '{:8.2f} mjd2000  '.format(self.t0)
        s += '-'.join([p[0] for p in self.seq]) + '  '
        s += '[' + ', '.join(['{:.2f}'.format(t) for t in self.tof]) + ']'
        return s
