# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 22:04:04 2020

@author: nikon
"""

# runfile for multinest stuff

# importing modules
import model_database as md
import implement_multinest as multi
from math import pi
import numpy as np

# IMPORTING DATA
data_1 = np.loadtxt("0_full_sim.txt", delimiter=",")
data_2 = np.loadtxt("6_full_sim.txt", delimiter=",")
data_3 = np.loadtxt("12_full_sim.txt", delimiter=",")
data_4 = np.loadtxt("18_full_sim.txt", delimiter=",")
freq = data_1[0]
signal_1 = data_1[1]
signal_2 = data_2[1]
signal_3 = data_3[1]
signal_4 = data_4[1]
signal = [signal_1, signal_2,signal_3, signal_4]
noise = 0.5e-2

# DEFINING MODEL
my_model = md.multi_fg_4(freq) # model selected from model_database.py

# DEFINING LOG LIKELIHOOD AND PRIORS
def log_likelihood(cube): # log likelihood function
    a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, c0, c1, c2, c3, c4, d0, d1, d2, d3, d4, amp, x0, width = cube
    foregrounds = my_model.foregrounds(cube)
    fg_1 = foregrounds[0]
    fg_2 = foregrounds[1]
    fg_3 = foregrounds[2]
    fg_4 = foregrounds[3]
    absorption = my_model.signal(cube) 
    model_1 = fg_1 + absorption
    model_2 = fg_2 + absorption
    model_3 = fg_3 + absorption
    model_4 = fg_4 + absorption
    normalise = 1/(np.sqrt(2*pi*noise**2)) 
    denominator = 2*noise**2
    numerator_1 = (signal[0] - model_1)**2 # likelihood depends on difference between model and observed temperature in each frequency bin
    numerator_2 = (signal[1] - model_2)**2 # likelihood depends on difference between model and observed temperature in each frequency bin
    numerator_3 = (signal[2] - model_3)**2 # likelihood depends on difference between model and observed temperature in each frequency bin
    numerator_4 = (signal[3] - model_4)**2 # likelihood depends on difference between model and observed temperature in each frequency bin
    loglike_1 = np.sum(np.log(normalise) - (numerator_1/denominator))
    loglike_2 = np.sum(np.log(normalise) - (numerator_2/denominator))
    loglike_3 = np.sum(np.log(normalise) - (numerator_3/denominator))
    loglike_4 = np.sum(np.log(normalise) - (numerator_4/denominator))
    return loglike_1 + loglike_2 + loglike_3 + loglike_4

def prior(cube): # priors for model parameters
   for i in range(20):
      cube[i]=-2000+2*2000*(cube[i])
   cube[10]=2*cube[10]
   cube[11]=100*cube[11]
   cube[12]=30*cube[12]
   return cube

multinest_object = multi.multinest_object(data=signal, model=my_model, priors=prior, loglike=log_likelihood)

if __name__ == "__main__":
    multinest_object.solve_multinest()