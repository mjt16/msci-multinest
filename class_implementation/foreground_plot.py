# A script to plot foregrounds

import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt("0_foreground_sim.txt",delimiter=",")
b = np.loadtxt("12_foreground_sim.txt",delimiter=",")

plt.figure()
plt.plot(a[0],a[1],'r-', label="0-12hr")
plt.plot(a[0],b[1],'b-', label="12-24hr")
plt.legend()
plt.savefig("foreground_comparison.png")
