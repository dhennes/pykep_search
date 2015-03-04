#!/usr/bin/env python

import matplotlib.pylab as plt

if __name__ == '__main__':
    
    # cassini
    X = [4, 8, 16, 32]
    Y = [744, 19837, 469123, 14196918]
    F = [12323.417874382676, 11538.534360909909, 9735.3954824788125, 6760.012894307366]
    
    # cassini_full
    X_full = [4, 8, 16, 32]
    Y_full = [1904, 87027, 4236400, 243105887]
    F_full = [12323.417874382676, 10405.482238107123, 8934.614192337669, 6571.4863484566258]

    fig = plt.figure() #plt.figure(figsize=(7,5))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(range(len(X)), Y, color='k')
    ax.plot(range(len(X_full)), Y_full, color='k', linestyle='--')
    ax.set_xlabel('Ephimeris grid resolution', fontsize=16)
    ax.set_ylabel('Lambert legs', fontsize=16)
    ax.set_yscale('log')

    ax.xaxis.set_ticks(range(len(X)))
    ax.set_xticklabels(['{:.2f}'.format(360./x).rstrip('0').rstrip('.') + '$^\circ$' for x in X])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(-0.5, len(X) - .5)
    
    # rosetta
    X = [4, 8, 16, 32]
    Y = [2489, 56019, 1388745, 42420347]
    F = [11163.647062715594, 10924.176555694088, 6498.0430788999847, 6498.0430789025195]

    # rosetta full
    X_full = [4, 8, 16, 32]
    Y_full = [5009, 226130, 12973175, 779191033]
    F_full = [11163.647062715594, 8782.4857375157226, 6128.5306927349666, 5887.1674194967718]

    # rosetta (correct emph)
#    X = [4, 8, 16, 32]
#    Y = [2696, 59787, 1538257, 47394078]
#    F = [8775.3026567814813, 7090.2708514845172, 6691.3341406862755, 5559.7144790522016]]]]
    
#    # rosetta full (correct emph)
#    X = [4, 8, 16, 32]
#    Y = [5466, 241851, 14794001, 895040579]
#    F = [7616.6281802875992, 7090.2708514845172, 6177.7904892994238, 4578.3397974254804]]]]
    
    ax.plot(range(len(X)), Y, color='k', marker='*')
    ax.plot(range(len(X_full)), Y_full, color='k', linestyle='--', marker='*')
    ax.set_xlabel('Ephimeris grid resolution', fontsize=16)
    ax.set_ylabel('Lambert legs', fontsize=16)
    ax.set_yscale('log')

    plt.legend(['Cassini - $t_0$ fixed', 'Cassini', 'Rosetta - $t_0$ fixed', 'Rosetta'], numpoints=1, loc='lower right')
    plt.savefig('paper/figures/complexity.eps', bbox_inches='tight')
    plt.show()
