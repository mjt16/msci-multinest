# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 00:14:51 2019

@author: matth
"""

# file to generate a set of noisy mock data according to a user-defined model

# importing modules
import model_database as md
import implement_multinest as multi
from math import pi
import numpy as np
import matplotlib.pyplot as plt

# fiducial model parameters
freq = np.linspace(51.965332, 97.668457, 50)
#a0 = 7.467527122442775
#a1 = -2.569712521825265
#a2 = -0.03731392522953101
#a3 = 0.049980346187051314
#a4 = 0.12055241703561778
a0 = 2625.771
a1 = -4202.081
a2 = 8636.317
a3 = -8954.631
a4 = 4553.795
a5 = -908.957
sim_coeffs = np.array([a0,a1,a2,a3,a4]) # simulated foreground coeffs
sim_amp = 0.057 # amplitude
sim_x0 = 78 # centre i.e. peak frequency
sim_width = 8.1 # width
sim_tau = 6.88 # optical depth
sim_phi = 5.74 
sim_l = 12.27
int_time = 1.6e8 # antennna integration time  
theta = [a0,a1,a2,a3,a4,sim_amp,sim_x0,sim_width]

mockmodel = md.logpoly_plus_gaussian(freq) # this is the user defined model, see model_database.py for more models
noise = np.random.normal(0, 10e-2, len(freq))

sim_signal = mockmodel.observation(theta) + noise

absorb = mockmodel.observation(theta, withFG=False) # seperating 21cm signal and foreground for plotting purposes
foregd = mockmodel.observation(theta, withSIG=False)

data = np.array([freq, absorb, foregd, noise, sim_signal])

np.savetxt("sim_signal_original.txt", data, delimiter=",") # save data in a .txt file; this mimics the procedure when actual data is imported e.g. EDGES

# plotting mock signal and noise
plt.subplot(1,2,1)
plt.plot(freq, sim_signal, 'r-')
plt.title("Simulated Signal")
plt.xlabel("Frequency/MHz")
plt.ylabel("Brightness Temperature/K")
plt.subplot(1,2,2)
plt.plot(freq, noise, 'bo')
plt.title("Simulated Noise")
plt.xlabel("Frequency/MHz")
plt.savefig("mockdata_original.png")
