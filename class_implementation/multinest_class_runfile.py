# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 00:14:51 2019

@author: matth
"""

# importing modules
import model_database as md
import implement_multinest as multi
from math import pi
import numpy as np

freq = np.linspace(51.965332, 97.668457, 50) # frequency array for whole procedure

# generating noisy data following a fiducial model, might want to move this code

# functions
def absorption(amp, x0, width): # signal 21cm absorption dip, defined as a negative gaussian
    return -amp*np.exp((-(freq-x0)**2)/(2*width**2))

def foreground(coeffs): # signal foreground
    freq_0 = 1 # SORT THIS OUT!!!
    l = len(coeffs)
    p = np.arange(0,l,1)
    freq_arr = np.transpose(np.multiply.outer(np.full(l,1), freq))
    normfreq = freq_arr/freq_0
    log_arr = np.log(normfreq)
    pwrs = np.power(log_arr, p)
    ctp = coeffs*pwrs
    log_t = np.sum(ctp,(1)) 
    temp = np.exp(log_t)
    return temp

def addnoise(data, int_time): # add noise to signal
    return data + data/(np.sqrt((freq[1]-freq[0])*(int_time)))

# fiducial model parameters
a0 = 7.467527122442775
a1 = -2.569712521825265
a2 = -0.03731392522953101
a3 = 0.049980346187051314
a4 = 0.12055241703561778
sim_coeffs = np.array([a0,a1,a2,a3,a4]) # simulated foreground coeffs
sim_amp = 0.5 # amplitude
sim_x0 = 78 # centre i.e. peak frequency
sim_width = 8.1 # width
int_time = 1.6e8 # antennna integration time  

simulated_clean = absorption(sim_amp, sim_x0, sim_width) + foreground(sim_coeffs) # simulated data without noise
simulated = addnoise(simulated_clean, int_time) # simulated data with noise
noise = simulated - simulated_clean # noise values

# DEFINING LOG LIKELIHOOD AND PRIORS
def log_likelihood(cube): # log likelihood function
    a0, a1, a2, a3, a4, amp, x0, width = cube#
    coeffs = [a0,a1,a2,a3,a4]
    model = absorption(amp, x0, width) + foreground(coeffs)
    normalise = 1/(np.sqrt(2*pi*noise**2))
    numerator = (simulated - model)**2 # likelihood depends on difference between model and observed temperature in each frequency bin
    denominator = 2*noise**2
    loglike = np.sum(np.log(normalise) - (numerator/denominator))
    return loglike
    
def prior(cube): # priors for model parameters
   for i in range(5):
      cube[i]=-10+2*10*(cube[i])
   cube[5]=cube[5]
   cube[6]=100*cube[6]
   cube[7]=20*cube[7]
   return cube

model = md.logpoly_plus_gaussian(freq) # model selected from model_database.py
multinest_object = multi.multinest_object(data=simulated, model=model, priors=prior, loglike=log_likelihood)

multinest_object.solve_multinest()
