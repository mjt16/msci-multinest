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
a0 = 7.467527122442775
a1 = -2.569712521825265
a2 = -0.03731392522953101
a3 = 0.049980346187051314
a4 = 0.12055241703561778
a5 = -0.09089525886322
sim_coeffs = np.array([a0,a1,a2,a3,a4,a5]) # simulated foreground coeffs
sim_amp = 0.057 # amplitude
sim_phi = 5.74 
sim_l = 12.27
int_time = 1.6e8 # antenna integration time  
theta = [a0,a1,a2,a3,a4,a5,sim_amp,sim_phi,sim_l]

mockmodel = md.hills_sine(freq) # this is the user defined model, see model_database.py for more models
noise = np.random.normal(0, 10e-2, len(freq))

sim_signal = mockmodel.observation(theta) + noise

absorb = mockmodel.observation(theta, withFG=False) # seperating 21cm signal and foreground for plotting purposes
foregd = mockmodel.observation(theta, withSIG=False)

data = np.array([freq, absorb, foregd, noise, sim_signal])

np.savetxt("sim_signal_hills.txt", data, delimiter=",") # save data in a .txt file; this mimics the procedure when actual data is imported e.g. EDGES

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
plt.savefig("mockdata_hills.png")
