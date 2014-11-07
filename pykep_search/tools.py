import numpy as np
import PyKEP as kep


PLANETS = {name: kep.planet_ss(name) for name in ['venus', 'earth', 'mars', 'jupiter', 'saturn']}


def enum(*sequential, **named):
    """Helper function to create enums."""
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)


def pretty_time(secs):
    """Returns a human readable time string."""
    hs, secs = divmod(secs, 60*60)
    mins, secs = divmod(secs, 60)
    return "%d:%02d:%04.1f" % (hs, mins, secs)    


def lambert_leg(p1, p2, t0, tof, vrel=None, dv_launch=0., rendezvous=False):
    """Compute a lambert leg from planet to planet.

    Arguments:
    p1 -- starting planet (str or PyKEP.planet object)
    p2 -- final planet (str or PyKEP.planet object)
    t0 -- start time of leg in MJD2000
    tof -- time of flight in days
    
    Keyword arguments:
    vrel -- caresian coordinates of the relative velocity before the flyby at p1
    dv_launch -- dv discounted at lunch (i.e. if vrel is None)
    rendezvous -- add final dv

    Returns:
    dV, vrel_out, where vrel_out is the relative velocity at the end of the leg at p2
    """
    # check if planets are names or planet objects
    if type(p1) is str:
        p1 = PLANETS[p1]
    if type(p2) is str:
        p2 = PLANETS[p2]
    r1, v1 = p1.eph(kep.epoch(t0))
    r2, v2 = p2.eph(kep.epoch(t0 + tof))
    lambert = kep.lambert_problem(r1, r2, tof * kep.DAY2SEC, p1.mu_central_body, False, 1)
    vrel_in = np.array(lambert.get_v1()[0]) - np.array(v1)
    vrel_out = np.array(lambert.get_v2()[0]) - np.array(v2)

    if vrel is None:
        # launch
        dv = max(np.linalg.norm(vrel_in) - dv_launch, 0)
    else:
        # flyby
        #print p1.name, p2.name, np.linalg.norm(vrel_in), np.linalg.norm(vrel_out)
        dv = kep.fb_vel(vrel, vrel_in, p1)

    if rendezvous:
        dv += np.linalg.norm(vrel_out)
        
    return dv, vrel_out


def jde_mga_1dsm(seq, t0, tof, slack=5, pop_size=50, n_evolve=10, dv_launch=6127., verbose=False):
    """Runs jDE with mga_1dsm problem."""
    from PyGMO.problem import mga_1dsm_tof
    from PyGMO.algorithm import jde
    from PyGMO import population

    prob = mga_1dsm_tof(seq=[kep.planet_ss(name) for name in seq],
                                   t0=[kep.epoch(t0-slack), kep.epoch(t0+slack)],
                                   tof=[[t-slack, t+slack] for t in tof],
                                   vinf=[0., dv_launch/1000.],
                                   add_vinf_arr=False)
    algo = jde(gen=500, memory=True)
    pop = population(prob, pop_size)
    if verbose:
        print pop.champion.f[0]
    for i in xrange(n_evolve):
        pop = algo.evolve(pop)
        if verbose:
            print pop.champion.f

            
# TODO: write verify function using archipelago 