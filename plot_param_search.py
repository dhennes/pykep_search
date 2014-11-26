#!/usr/bin/env python

import os
import sys
import pickle
import numpy as np
import scipy
import scipy.stats
import matplotlib.pylab as plt

if __name__ == '__main__':
    fnames = [os.path.join(dp, f) for dp, dn, filenames in os.walk('results') for f in filenames if os.path.splitext(f)[1] == '.pkl']

    for fname in [f for f in fnames if not 'perf' in f]:
        print fname,

        data = pickle.load(open(fname, 'rb'))
        print len(data)
        dv, c_P = zip(*data)
        dv = np.array(dv)
        c_P = np.array(c_P)

        MIN_DV = 6760.012894307366 # cassini fixed t0
        if 'full' in fname:
            MIN_DV = 6571.4863484566258 # cassini full

        if 'rosetta' in fname:
            MIN_DV =  6498.0430789025195 # rosetta fixed t0
            if 'full' in fname:
                MIN_DV = 5887.17
        # 100m/s slack
        MIN_DV += 50.
            
        
        params = None
        min_idx = np.where(dv <= MIN_DV)
        samples = c_P[min_idx]
        if len(min_idx[0]) >= 10:
            params=scipy.stats.lognorm.fit(samples, floc=0)
            print params
            
        
        fig = plt.figure() #figsize=(5,5))
        ax = fig.add_subplot(1,1,1)
      
        ax.scatter(c_P, dv, 4, 'k', edgecolor='None')
        ax.scatter(c_P[min_idx], dv[min_idx], 4, 'r', edgecolor='None')
        ax.set_xlim(0.001, 10)
        ax.set_ylim(5000, 14000)
        if 'egreedy' in fname:
            ax.set_xlabel('$\epsilon$', fontsize=16)
        else:
            ax.set_xlabel('$C_p$', fontsize=16)
            
        ax.set_ylabel('$\Delta v$ [km/s]', fontsize=16)
        ax.set_xscale('log')
        labels = ['{:.1f}'.format(x/1000.) for x in ax.yaxis.get_majorticklocs()]
        labels[0] = ''
        ax.set_yticklabels(labels)
        ax.tick_params(axis='both', which='major', labelsize=13)

        import matplotlib.ticker
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.3f}'.format(x).rstrip('0').rstrip('.')))


        ax2 = ax.twinx()
        ax2.axis('off')
        ax.set_zorder(ax2.get_zorder()+1)
        ax.patch.set_visible(False)
        ax2.set_ylim(0., 100.)

        if not params is None:
            x = np.logspace(-3, 1, num=1000)
            y = scipy.stats.lognorm.pdf(x, params[0], loc=params[1], scale=params[2])# * 10 + 5000 
            y = y * params[2] * 100./9.
            ax2.plot(x, y, 'k')

            mean = scipy.stats.lognorm.stats(params[0], loc=params[1], scale=params[2], moments='m')
            print 'mean: %.4f' % mean
            #ax2.vlines([params[2]], 0., 100., 'k', linestyle=':')
            ax2.vlines([mean], 0., 100., 'k', linestyle=':')

        if len(samples) > 0:
            n, bins, patches = ax2.hist(samples, bins=10 ** np.linspace(-3, 1, num=100), normed=False)
            for p in patches:
                p.set_facecolor([.75] * 3)
                p.set_edgecolor([.75] * 3)

        ofname = 'paper/figures/' + '/'.join(fname.split('/')[1:]).split('.')[0][:-3] + 'K.eps'
        dirname = '/'.join(ofname.split('/')[:-1])
        ofname = ofname.replace('param_search', dirname.split('/')[-1])
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print ofname
        plt.savefig(ofname, bbox_inches='tight')
    
        #plt.show()
