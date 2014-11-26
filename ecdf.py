#!/usr/bin/env python
#from rat import Rat
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle

def get_cut_point(fevals, max_fevals=1e7):
    """ get cutting point to trim trajectory to max_fevals """
    nonz = (fevals >= max_fevals).nonzero()[0]
    if len(nonz) > 0:
        return min(nonz)
    else:
        return None

def cut_converged(fevals):
    last = 0
    for i, f in enumerate(fevals):
        if f == last:
            return i-1
        last = f
    return None

def mono_dec(x):
    """ transforms input to monotonically decreasing vector """
    for i in xrange(1, len(x)):
        if x[i] > x[i-1]:
            x[i] = x[i-1]
    return x

if __name__ == "__main__":
    # target value
    ft = 7000
    # target delta
    df = 100

#    # problem dimensions
#    dim = 19


    table = []
    legendstr = []
    X = []
    Y = []

    max_fevals = int(1e7)

    
    exps = ['results/perf/egreedy_50000.pkl']
    
    for exp in exps:
        print "-"*80
        print exp
        print "target:\t%f + %f"%(ft, df)
        print "max-feval:\t%d"%max_fevals

        results = pickle.load(open(exp, 'rb'))

        # arrays for trails
        F = []
        Fevals = []
        runs_used = []

        current_f = []
        current_fevals = []
        # assemable collumns

        print "iterations:\t%d"%len(results)


        something = np.array([r[0] for r in results])
        print something.size
        print "average fevals: %f"%np.mean(something)

        
        for i, r in enumerate(results):
            # append r to current


            f = r[0]
            fevals = np.array(r[1])
            
            # # cut when converged / this just saves time
            # cut = cut_converged(fevals)
            # if cut is not None:
            #     f = f[0:cut]
            #     fevals = fevals[0:cut]

            current_f.extend(f)

            if len(current_fevals) > 0:
                fevals = fevals + current_fevals[-1]
            current_fevals.extend(fevals)

            cut = get_cut_point(np.array(current_fevals), max_fevals=max_fevals)

            if cut is not None:
                F.append(mono_dec(current_f[0:cut]))
                Fevals.append(current_fevals[0:cut])
                current_fevals = []
                current_f = []
                runs_used.append(i+1)

        print "valid trails:\t%d"%len(F)

        n = 100
        F = F[0:n]
        n = min(n, len(F))
        perc = []
        print "used trails:\t%d"%n
        print "used runs:\t%d"%runs_used[n-1]
        if n > 0:
            feval_conv = []
            for (f, feval) in zip(F, Fevals):
                # find convergence point
                feval = np.array(feval)
                f = np.array(f)
                nonz = (f < ft + df).nonzero()[0]
                if nonz.size > 0:
                    # add convergence point            
                    feval_conv.append(feval[min(nonz)])
            

            perc = np.array(range(1,len(feval_conv)+1))/float(n)
            X.append(sorted(feval_conv))
            Y.append(perc)
            algo_name = exp
            algo_name = exp.replace("-", " ").replace("jde", "jDE").replace("cmaes", "CMA-ES").replace("mbh snopt", "MBH-SNOPT").replace("death penalty ", "NP=").replace("co evolution", "co-evolution")
            legendstr.append(algo_name)
            p_s = perc[-1]
            ET_s = np.mean(feval_conv)
            SP2 = (1-p_s)/p_s*max_fevals + ET_s
            print "total convered at max_fevals: %.2f"%p_s
            print "average feval for successful: %f"%ET_s
            print "SP2: %f"%SP2

            table.append((p_s, "%s \t&\t %.2f \t&\t %s \t&\t %s \\\\"%(algo_name, 
                                                                       p_s, 
                                                                       "{:,}".format(int(ET_s)), 
                                                                       "{:,}".format(int(SP2)))))



    # print sorted latex table
    _, table = zip(*sorted(table, key=lambda (x, y): x, reverse=True)) 
    print "="*80
    print "\n".join(table)                     

    plt.axes().set_xscale("log")
    plt.axes().set_xlim([10e3, max_fevals])
    plt.axes().set_ylim([0, 1])
    plt.grid(b=True, which='major', linestyle='--')

    legendstr, X, Y = zip(*sorted(zip(legendstr, X, Y), key=lambda (tmp1, tmp2, x): x[-1], reverse=True))

    color = ["k"] * 4 + [(.5, .5, .5)] * 4
    linestyle = ["-", ":", "--", "-."] * 2
    for x, y, c, ls in zip(X,Y, color, linestyle):
        x = np.concatenate(([0], x, [max_fevals]))
        y = np.concatenate(([0], y, [y[-1]]))
        plt.step(x,y, where="post", color=c, linestyle=ls, linewidth=1.5)

    # add last data point for label
    #X = np.append(X, 10*X[-1])
    #for i, data in enumerate(Y):
        #data.append(data[-1])
    #    plt.step(X, data)
        #plt.text(X[-1]/10, data[-1], legendstr[i]) 


    #plt.axis((0, max(feval), 0, 1))
    # # x0, x1 = plt.axes().get_xlim()
    # plt.axes().set_ylim([0, 1])
    # plt.axes().set_xlim([0, max(feval)])

    # # plt.axes().set_aspect((x1 - x0)/1)


    plt.xlabel("function evaluations")
    plt.ylabel("proportion of converged trials")    
    legend = plt.legend(legendstr, loc="upper left")
    for obj in legend.legendHandles:
        obj.set_linewidth(1.5)
#    plt.show()

#    fname = "result.eps"
#    plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
#                orientation='portrait', papertype=None, format="eps",
#                transparent=True, bbox_inches=None, pad_inches=0.1,
#                frameon=None)

