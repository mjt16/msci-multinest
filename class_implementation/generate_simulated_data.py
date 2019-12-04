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
import matplotlib.pyplot as plt  #this should be moved 

freq = np.linspace(51.965332, 97.668457, 50) # frequency array for whole procedure

# generating noisy data following a fiducial model, might want to move this code

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
    temp = np.exp(log_t)
    return temp

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

clean = absorption(sim_amp, sim_x0, sim_width) + foreground(sim_coeffs)
noise = np.random.normal(0, 10e-2, len(freq)) #add noise

sim_signal = clean + noise

data = np.array([freq, sim_signal, noise])

np.savetxt("sim_signal.txt", data, delimiter=",")

#plotting
plt.subplot(1,2,1)
plt.plot(freq, sim_signal, 'r-')
plt.title("Simulated Signal")
plt.xlabel("Frequency/MHz")
plt.ylabel("Brightness Temperature/K")
plt.subplot(1,2,2)
plt.plot(freq, noise, 'bo')
plt.title("Simulated Noise")
plt.xlabel("Frequency/MHz")
plt.ylabel("Brightness Temperature/K")
plt.savefig("subplots.png")
