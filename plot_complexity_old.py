#!/usr/bin/env python

import matplotlib.pylab as plt

if __name__ == '__main__':
#    X = [4, 8, 12, 16, 20, 24, 28, 32, 36]
#    Y = [744, 19837, 144422, 593708, 1906603, 5104174, 12076773, 25804568, 50339677]
#    F = [12323.417874382676, 11538.534360909909, 9530.1638666779654, 9735.3954824788125, 7703.8159139919962, 8981.3920004147931, 7259.1668005152369, 6760.012894307366, 8046.5281153442693]

    #X = [4, 8, 12, 16, 20, 24, 28, 32, 36]
    #Y = [744, 19837, 144422, 593708, 1906603, 5104174, 12076773, 25804568, 50339677]
    #F = [12323.417874382676, 11538.534360909909, 9530.1638666779654, 9530.1638666779654, 7703.8159139919962, 7703.8159139919962, 7259.1668005152369, 6760.012894307366, 6760.012894307366]

    #Y_full = [1904, 87027, 826879, 4976252, 19230024, 64458196, 176536100, 415405587]
    #F_full = [12323.417874382676, 10405.482238107123, 9530.1638666779654, 8934.614192337669, 7703.8159139919962, 7703.8159139919962, 6746.8112872864594, 6571.4863484566258]

    #Y_full = [1904, 87027, 826879, 4976252, 19230024, 64458196, 176536100, 415405587, 897319492]
    #F_full = [12323.417874382676, 10405.482238107123, 9530.1638666779654, 8934.614192337669, 7703.8159139919962, 7703.8159139919962, 6746.8112872864594, 6571.4863484566258, 6571.4863484566258]

    
#    # rosetta
#    X = [4, 8, 16, 32]
#    Y = [2489, 56019, 1388745, 42420347]
#    F = [11163.647062715594, 10924.176555694088, 6498.0430788999847, 6498.0430789025195]


#    # rosetta full / incomplete
#    X_full = [4, 8, 16, 32]
#    Y_full = [5009, 226130, 12973175]
#    F_full = [11163.647062715594, 8782.4857375157226, 6128.5306927349666]
    
    # cassini
    X = [4, 8, 16, 32]
    Y = [744, 19837, 469123, 14196918]
    F = [12323.417874382676, 11538.534360909909, 9735.3954824788125, 6760.012894307366]
    
    # cassini_full
    X_full = [4, 8, 16, 32]
    Y_full = [1904, 87027, 4236400, 243105887]
    F_full = [12323.417874382676, 10405.482238107123, 8934.614192337669, 6571.4863484566258]

    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.set_ylim(5000, 13000)
    
    ax1.scatter(range(len(X)), F, marker='*', color='k')
    ax1.scatter(range(len(X_full)), F_full, 60, marker='o', color='k', facecolors='none')
    ax1.set_xlabel('Ephimeris grid resolution', fontsize=16)
    ax1.set_ylabel('$\Delta v$ [km/s]', fontsize=16)
    ax1.set_yticklabels(['{:.1f}'.format(x/1000.) for x in ax1.yaxis.get_majorticklocs()])
    ax1.tick_params(axis='both', which='major', labelsize=13)


    
    ax2 = ax1.twinx()
    ax2.plot(range(len(X)), Y, color='k') #, marker='o')
    ax2.plot(range(len(X_full)), Y_full, color='k', linestyle=':')#, marker='*')
    ax2.set_xlabel('Ephimeris grid resolution', fontsize=16)
    ax2.set_ylabel('Lambert legs', fontsize=16)
    ax2.set_yscale('log')

    ax2.xaxis.set_ticks(range(len(X)))
    ax2.set_xticklabels(['{:.2f}'.format(360./x).rstrip('0').rstrip('.') + '$^\circ$' for x in X])
    ax2.tick_params(axis='both', which='major', labelsize=14)

    ax2.set_xlim(-0.5, len(X) - .5)
    
    plt.savefig('paper/figures/complexity.eps', bbox_inches='tight')
    
    plt.show()
