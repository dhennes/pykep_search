import copy
import numpy as np
import bisect

import PyKEP as kep
import tools



PLANET_TOF = {}
PLANET_TOF[('earth', 'venus')] = (30., 400.)
PLANET_TOF[('earth', 'earth')] = (340., 800.)
PLANET_TOF[('earth', 'mars')] = (100., 1000.)
PLANET_TOF[('earth', 'jupiter')] = (400., 1600.)
#PLANET_TOF[('earth', 'saturn')] = (1200., 2500.)
PLANET_TOF[('earth', '67p')] = (500., 3000.)


PLANET_TOF[('venus', 'venus')] = (224.7, 450.)
PLANET_TOF[('venus', 'mars')] = (100., 600.)
PLANET_TOF[('venus', 'jupiter')] = (400., 1600.)
#PLANET_TOF[('venus', 'saturn')] = (1200., 2500.)
PLANET_TOF[('venus', '67p')] = (500., 3000.)

PLANET_TOF[('mars', 'mars')] = (780., 1600.)
PLANET_TOF[('mars', 'jupiter')] = (400., 1600.)
#PLANET_TOF[('mars', 'saturn')] = (1200., 2500.)
PLANET_TOF[('mars', '67p')] = (500., 3000.)

PLANET_TOF[('jupiter', 'jupiter')] = (4332., 9000.)
#PLANET_TOF[('jupiter', 'saturn')] = (800., 2500.)
PLANET_TOF[('jupiter', '67p')] = (500., 3000.)



for (p1, p2) in PLANET_TOF.keys():
    PLANET_TOF[(p2, p1)] = PLANET_TOF[(p1, p2)]

chury = kep.planet(kep.epoch(2456879.5, kep.epoch.epoch_type.JD),
                   (
                       3.4630 * kep.AU, # a
                       0.64102, # e
                       7.0405 * kep.DEG2RAD, # i
                       50.147 * kep.DEG2RAD, # W
                       12.780 * kep.DEG2RAD, # w
                       303.71 * kep.DEG2RAD # M
                   ),
                   kep.MU_SUN,
                   1e-10, # mu_self
                   5.e3, # radius
                   100.e3, # save_radius
                   '67p'
)
tools.PLANETS[chury.name] = chury

for p in tools.PLANETS.values():
    if not p.name is 'jupiter':
        p.save_radius = 1.05

    
PLANET_NAMES = ['venus', 'earth', 'mars', 'jupiter', '67p'] #, 'jupiter', 'saturn']

T0 = (1460., 1825.) #(5480., 5480.+365.25) # # launch window
T_MIN = T0[0]
T_MAX = T0[-1] + 4000.

def set_t_res(t_res):
    global T_RES
    global T_SCALE
    T_RES = t_res
    T_SCALE = {name: np.arange(T_MIN, T_MAX, tools.PLANETS[name].period/kep.DAY2SEC/T_RES) for name in PLANET_NAMES}

set_t_res(32)
#print '\n'.join('time grid - %s: %d' % (name, len(T_SCALE[name])) for name in PLANET_NAMES)

MAX_EPOCH = T_SCALE['67p'][-2]
MAX_FLYBYS = 5
MAX_DV = 10000.
MAX_DV_LAUNCH = 5000.
MOVE_TYPE = tools.enum('T0', 'PLANET', 'TOF')
MOVES = {
    MOVE_TYPE.T0: [t for t in T_SCALE['earth'] if t >= T0[0] and t <= T0[1]], 
    MOVE_TYPE.PLANET: PLANET_NAMES,
    MOVE_TYPE.TOF: None, # defined on the fly
}


def fix_first_move(fixit):
    if fixit:
        MOVES[MOVE_TYPE.T0] = [min(T_SCALE['earth'], key=lambda x: abs(1550.-x))]
    else:
        MOVES[MOVE_TYPE.T0] = [t for t in T_SCALE['earth'] if t >= T0[0] and t <= T0[1]]
        
    
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
        if self.isterminal():
            return []
        if self.next_move is MOVE_TYPE.TOF:
            min_tof, max_tof = PLANET_TOF[self.seq[-2], self.seq[-1]]
            cur_t = self.t0 + sum(self.tof)
            lb = bisect.bisect(T_SCALE[self.seq[-1]], cur_t + min_tof)
            ub = bisect.bisect(T_SCALE[self.seq[-1]], cur_t + max_tof)
            #ub = min(ub, lb + 50)
            return [t - cur_t for t in T_SCALE[self.seq[-1]][lb:ub]]
        else:
            return MOVES[self.next_move]


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
#            if not self.isterminal():
#                self.random_move() # TODO: WOW don't use this
            return
        elif self.next_move == MOVE_TYPE.TOF:
            self.tof.append(move)
            self.next_move = MOVE_TYPE.PLANET
            # evaluate lambert leg
            t = self.t0 + sum(self.tof[:-1])
            tof = self.tof[-1]
            dv, vrel_out = tools.lambert_leg(self.seq[-2], self.seq[-1], t, tof,
                                             vrel=self.vrel,
                                             rendezvous=self.isfinal())

            if len(self.tof) == 1:
                dv = max(dv - MAX_DV_LAUNCH, 0)
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
        if len(self.tof) == MAX_FLYBYS:
            return True
        if self.dv is not None and self.dv > MAX_DV:
            return True
        if self.t0 is not None and self.t0 + sum(self.tof) > MAX_EPOCH:
            return True
        if self.isfinal():
            return True
        return False


    def isfinal(self):
        return self.seq[-1] == '67p' and len(self.seq) -1 == len(self.tof)
        

    def get_value():
        return self.dv

        
    def copy(self):
        return State(seq=self.seq, t0=self.t0, tof=self.tof, vrel=self.vrel, dv=self.dv, next_move=self.next_move)

    def __key(self):
        return (self.seq, self.t0, self.tof)


    def __eq__(s1, s2):
        return s1.__key() == s2.__key()


    def __hash__(self):
        return hash(self.__key())
        
        
    def __repr__(self):
        s = '{:8.2f} m/s  '.format(self.dv)
        s += '{:7.2f} days  '.format(sum(self.tof))
        if self.t0 is not None:
            s += '{:8.2f} mjd2000  '.format(self.t0)
        s += '-'.join([p[0] for p in self.seq]) + '  '
        s += '[' + ', '.join(['{:.2f}'.format(t) for t in self.tof]) + ']'
        return s

        
if __name__ == '__main__':
    seq = ['earth', 'earth', 'mars', 'earth', 'earth', '67p']
    #print T_SCALE["saturn"]
    
    s = State()
    s.move(min(s.moves(), key=lambda move: abs(move-x[0]))) # t0
    for (p, tof) in zip(seq[1:], x[1:]):
        s.move(p)
        print tof
        print s.moves()
        s.move(min(s.moves(), key=lambda move: abs(move-tof)))
        #s.move(tof)
        print s
    
