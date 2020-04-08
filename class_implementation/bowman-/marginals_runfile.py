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

n_params = 9

prefix = 'bowman-'

an = Analyzer(n_params, outputfiles_basename=prefix)
marge = plot.PlotMarginalModes(an)
plt.figure()
"""
for i in [-1,-2,-3,-4]:
    plt.subplot(1,4,abs(i))
    out = marge.plot_conditional(i)
plt.savefig("output.png")
"""
out = marge.plot_conditional(5)
plt.savefig("output.png")
