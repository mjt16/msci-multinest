# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 00:14:51 2019

@author: matth
"""

# runfile for multinest stuff

# importing modules
import model_database as md
import implement_multinest as multi
from math import pi
import numpy as np

# IMPORTING SIMULATED DATA
data = np.loadtxt("sim_signal.txt", delimiter=",")
freq = data[0]
sim_signal = data[1]
noise = data[2]

# DEFINING MODEL
my_model = md.logpoly_plus_gaussian(freq) # model selected from model_database.py

# DEFINING LOG LIKELIHOOD AND PRIORS
def log_likelihood(cube): # log likelihood function
    a0, a1, a2, a3, a4, amp, x0, width = cube
    model = my_model.observation(cube)
    normalise = 1/(np.sqrt(2*pi*noise**2))
    numerator = (sim_signal - model)**2 # likelihood depends on difference between model and observed temperature in each frequency bin
    denominator = 2*noise**2
    loglike = np.sum(np.log(normalise) - (numerator/denominator))
    return loglike
    
def prior(cube): # priors for model parameters
   for i in range(5):
      cube[i]=-10+2*10*(cube[i])
   cube[5]=-1 + 2*cube[5]
   cube[6]=100*cube[6]
   cube[7]=20*cube[7]
   return cube

multinest_object = multi.multinest_object(data=sim_signal, model=my_model, priors=prior, loglike=log_likelihood)

multinest_object.solve_multinest()

