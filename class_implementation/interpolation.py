# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:17:36 2020

@author: nikon
"""

from scipy.interpolate import CubicSpline
import numpy as np

# import data
data = np.loadtxt("12_full_sim.txt", delimiter=",")
freq = data[0]
signal = data[1]
   

cs = CubicSpline(freq, signal,bc_type='natural') # interpolate using a cubic spline

new_freq = np.arange(50, 99, 0.25)
new_sig = np.zeros_like(new_freq)

for i in range(len(new_freq)):
    new_sig[i] = cs(new_freq[i])
    
data = np.array([new_freq, new_sig])

np.savetxt("12_full_sim_interpol.txt", data, delimiter=",")
    

    