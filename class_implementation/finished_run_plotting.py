#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
__doc__ = """
Script that does default visualizations (marginal plots, 1-d and 2-d).

Author: Johannes Buchner (C) 2013-2019
"""
import numpy
from numpy import exp, log
import matplotlib.pyplot as plt
import sys, os
import json
import pymultinest
import corner
import model_database as md

if len(sys.argv) != 2:
	sys.stderr.write("""SYNOPSIS: %s <output-root> 

	output-root: 	Where the output of a MultiNest run has been written to. 
	            	Example: chains/1-
%s""" % (sys.argv[0], __doc__))
	sys.exit(1)

prefix = sys.argv[1]
print('model "%s"' % prefix)
if not os.path.exists(prefix + 'params.json'):
	sys.stderr.write("""Expected the file %sparams.json with the parameter names.
For example, for a three-dimensional problem:

["Redshift $z$", "my parameter 2", "A"]
%s""" % (sys.argv[1], __doc__))
	sys.exit(2)
parameters = json.load(open(prefix + 'params.json'))
n_params = len(parameters)

a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = prefix)
s = a.get_stats()

json.dump(s, open(prefix + 'stats.json', 'w'), indent=4)

print('  marginal likelihood:')
print('    ln Z = %.1f +- %.1f' % (s['global evidence'], s['global evidence error']))
print('  parameters:')
paramlist = [] # for plotting model vs data
for p, m in zip(parameters, s['marginals']):
	lo, hi = m['1sigma']
	med = m['median']
	paramlist.append(med)
	sigma = (hi - lo) / 2
	if sigma == 0:
		i = 3
	else:
		i = max(0, int(-numpy.floor(numpy.log10(sigma))) + 1)
	fmt = '%%.%df' % i
	fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
	print(fmts % (p, med, sigma))

print('creating marginal plot ...')
data = a.get_data()[:,2:]
weights = a.get_data()[:,0]

#mask = weights.cumsum() > 1e-5
mask = weights > 1e-4

corner.corner(data[mask,:], weights=weights[mask], 
	labels=parameters, show_titles=True)
plt.savefig(prefix + 'corner.pdf')
plt.savefig(prefix + 'corner.png')
plt.close()

# IMPORTING MOCK DATA
sim_stuff=numpy.loadtxt("sim_signal.txt", delimiter=",")
freq = sim_stuff[0]
sim_signal = sim_stuff[4]
noise = sim_stuff[3]
absorb = sim_stuff[1]

# GETTING CONVERGED MODEL
model_signal = md.logpoly_plus_gaussian(freq)
final_vals = numpy.array(paramlist)

# PLOTTING MOCK DATA VS. CONVERGED MODEL
plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.plot(freq, sim_signal, 'ro', label="mock")
plt.plot(freq, model_signal.observation(final_vals), 'b-', label="model")
plt.legend()
plt.title("Model vs. Data (full range)")
plt.xlabel("Frequency/MHz")
plt.ylabel("Brightness Temperature/K")
plt.subplot(1,3,2)
plt.plot(freq, absorb, 'ro', label="mock")
plt.plot(freq, model_signal.observation(final_vals, withFG=False), 'b-', label="model")
plt.legend()
plt.title("Model vs. Data (21cm only)")
plt.xlabel("Frequency/MHz")
plt.subplot(1,3,3)
plt.plot(freq, sim_signal-model_signal.observation(final_vals), 'b-')
plt.title("Residuals (full range)")
plt.xlabel("Frequency/MHz")
plt.savefig("model_vs_data.png", dpi=200)
