# A script to add the 21cm absorption

# Importing modules
import numpy as np
import matplotlib.pyplot as plt

# Foreground arrays (no 21cm feature)
a = np.loadtxt("0_foreground_1w_sim.txt", delimiter=",")
#b = np.loadtxt("6_foreground_4w_sim.txt", delimiter=",")
#c = np.loadtxt("12_foreground_4w_sim.txt", delimiter=",")
#d = np.loadtxt("18_foreground_4w_sim.txt", delimiter=",")

freq = a[0]

# Defining 21cm feature
amp = 0.05
x0 = 78
width = 8.06
t21 = -amp*np.exp((-(freq-x0)**2)/(2*width**2))

# Adding feature to foregrounds
a[1] = a[1] + t21
#b[1] = b[1] + t21
#c[1] = c[1] + t21
#d[1] = d[1] + t21

# Saving to .txt
np.savetxt("0_full_1w_sim.txt",a,delimiter=",")
#np.savetxt("6_full_4w_sim.txt",b,delimiter=",")
#np.savetxt("12_full_4w_sim.txt",c,delimiter=",")
#np.savetxt("18_full_4w_sim.txt",d,delimiter=",")

# Plotting to check
plt.figure()
plt.subplot(1,2,1)
plt.plot(freq,a[1],"r-",label="0-24hr foreground + 21cm feature")
#plt.plot(freq,b[1],"b-",label="6-12hr foreground + 21cm feature")
#plt.plot(freq,c[1],"g-",label="12-18hr foreground + 21cm feature")
#plt.plot(freq,d[1],"m-",label="18-24hr foreground + 21cm feature")
plt.legend()
plt.subplot(1,2,2)
plt.plot(freq,t21,"r-")
plt.savefig("test_graphs_1w.png")
