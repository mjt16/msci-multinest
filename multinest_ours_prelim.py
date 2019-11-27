# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:34:02 2019

@author: mjt16
"""

# nested sampling script

#importing modules
from __future__ import absolute_import, unicode_literals, print_function
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from pymultinest.solve import solve
#from pymultinest.run import run
import os
try: os.mkdir('chains')
except OSError: pass

# define frequency array
freq = np.linspace(51.965332, 97.668457, 50)

# functions
def absorption(amp, x0, width): # signal 21cm absorption dip, defined as a negative gaussian
    return -amp*np.exp((-(freq-x0)**2)/(2*width**2))

def foreground(coeffs): # signal foreground
    freq_0 = 78 # SORT THIS OUT!!!
    l = len(coeffs)
    p = np.arange(0,l,1)
    freq_arr = np.transpose(np.multiply.outer(np.full(l,1), freq))
    normfreq = freq_arr/freq_0
    log_arr = np.log(normfreq)
    pwrs = np.power(log_arr, p)
    ctp = coeffs*pwrs
    log_t = np.sum(ctp,(1)) 
    return np.exp(log_t)

def addnoise(data, int_time): # add noise to signal
    return data + data/(np.sqrt((freq[1]-freq[0])*(int_time)))

# creating simulated foreground data using EDGES foreground; see edgesfit.py
a0 = 7.467527122442775
a1 = -2.569712521825265
a2 = -0.03731392522953101
a3 = 0.049980346187051314
a4 = 0.12055241703561778
sim_coeffs = np.array([a0,a1,a2,a3,a4]) # simulated foreground coeffs

# creating simulated 21cm data
sim_amp = 0.5 # amplitude
sim_x0 = 78 # centre i.e. peak frequency
sim_width = 8.1 # width

int_time = 1.6e8 # antennna integration time  

simulated_clean = absorption(sim_amp, sim_x0, sim_width) + foreground(sim_coeffs)
simulated = addnoise(simulated_clean, int_time)
noise = simulated - simulated_clean
print(noise)
def log_likelihood(cube): # log likelihood function for model parameters theta, simulated data, and model data
    a0, a1, a2, a3, a4, amp, x0, width = cube # theta takes form of array of model parameters
    coeffs = [a0,a1,a2,a3,a4]
    model = absorption(amp, x0, width) + foreground(coeffs)
    normalise = 1/(np.sqrt(2*pi*noise**2))
    numerator = (simulated - model)**2 # likelihood depends on difference between model and observed temperature in each frequency bin
    denominator = 2*noise**2
    #lj = normalise*np.exp(-numerator/denominator)
    return np.sum(np.log(normalise) - (numerator/denominator)) 
    #return np.sum(np.log(lj)) # sum over all frequency bins

def prior(cube):
   for i in range(5):
      cube[i]=-10+20*(cube[i])
   cube[5]=cube[5]
   cube[6]=100*cube[6]
   cube[7]=20*cube[7]
   """
   for i in range(5,8):
      if i==6:
         cube[i]=(100*cube[i])
      else:
         cube[i]=200*cube[i]
   """
   """ 
   cube[0]=-1000 + 2*10**cube[0]*3)
   cube[1]=-4*10**5 + 2*4*10**(cube[1]*4)
   cube[2]=-5*10**2 + 2*5*10**(cube[2]*2)
   cube[3]=-10**2 + 2*10**(cube[3]*2)
   cube[4]=-10**2 + 2*10**(cube[4]*2)
   cube[5]=-10**3 + 2*10**(cube[5]*3)
   cube[6]=-1 + 10**(cube[6]*3)
   cube[7]=-1 + 10***(cube[7]*3)
   """   
   return cube

# number of dimensions our problem has
parameters = ["a0", "a1", "a2", "a3", "a4", "amp", "x0", "width"]
n_params = len(parameters)
# name of the output files
prefix = "testoutput1-"

# run MultiNest
result = solve(LogLikelihood=log_likelihood, Prior=prior, 
	n_dims=n_params, outputfiles_basename=prefix, verbose=True)
"""
result = run(LogLikelihood=log_likelihood, Prior=prior, n_dims=n_params, n_live_points=1000, outputfiles_basename=prefix, resume=True, verbose=True)
"""
print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

# make marginal plots by running:
# $ python multinest_marginals.py chains/3-
# For that, we need to store the parameter names:
import json
with open('%sparams.json' % prefix, 'w') as f:
	json.dump(parameters, f, indent=2)
