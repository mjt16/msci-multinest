# a script to produce marginalised parameters

# importing modules
from __future__ import absolute_import, unicode_literals, print_function
from pymultinest.solve import solve
from pymultinest.analyse import Analyzer
import os
import sys
import shutil
import plot
import matplotlib.pyplot as plt
try: os.mkdir('chains')
except OSError: pass

do_pretty = True

n_params = 23

prefix = 'multi_fg_4-'

an = Analyzer(n_params, outputfiles_basename=prefix)
for i in range(n_params):
    plt.figure()
    marge = plot.PlotMarginalModes(an)
    if do_pretty == True:
        out = marge.plot_modes_marginal(i,with_ellipses=False)
        plt.savefig("parameter%i_marginal.png"%i)
    else:
        out = marge.plot_conditional(i,with_ellipses=False)
        plt.savefig("posterior%i_marginal.png"%i)

