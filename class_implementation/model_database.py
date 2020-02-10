# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:37:16 2019

@author: matth
"""

# importing modules
import base_model_class as bmc
import numpy as np

# DATABASE OF DIFFERENT FOREGROUND AND 21CM MODELS ============================
class logpoly_plus_gaussian(bmc.model):
    """
    A log polynomial foreground up to 4th order
    and a gaussian absorption for 21cm signal

    Requires parameters in form
    theta = [a0,a1,a2,a3,a4,amp,x0,width]
    """
    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "log_poly_4"
        self.name_sig = "gaussian"
        self.labels = ["a0","a1","a2","a3","a4","amp","x0","width"]
        pass

    def foreground(self, theta):
        """
        Log polynomial foreground up to 4th order
        """
        freq_0 = 75 # SORT THIS OUT!!! pivot scale
        coeffs = theta[0:-3]
        l = len(coeffs)
        p = np.arange(0,l,1)
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        log_arr = np.log(normfreq)
        pwrs = np.power(log_arr, p)
        ctp = coeffs*pwrs
        log_t = np.sum(ctp,(1))
        fg = np.exp(log_t)
        return fg

    def signal(self, theta): # signal 21cm absorption dip, defined as a negative gaussian
        amp = theta[-3]
        x0 = theta[-2]
        width = theta[-1]
        t21 = -amp*np.exp((-(self.freq-x0)**2)/(2*width**2))
        return t21

# =============================================================================

class bowman(bmc.model):
    """
    Model used by Bowman in 2018 paper eq.2

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, amp, x0, width, tau]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial"
        self.name_sig = "flat gaussian"
        self.labels = ["a0","a1","a2","a3","a4","amp","x0","width","tau"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 4th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-4]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-4]
        x0 = theta[-3]
        width = theta[-2]
        tau = theta[-1]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

# =============================================================================

class bowman_physical(bmc.model):
    """
    Model used by Bowman in 2018 paper eq.3

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, amp, x0, width, tau]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "physical polynomial"
        self.name_sig = "flat gaussian"
        self.labels = ["a0","a1","a2","a3","a4","amp","x0","width","tau"]
        pass

    def foreground(self, theta):
        """
        Physical foreground model
        """
        freq_0 = 75.0
        coeffs = theta[0:-4]
        normfreq = self.freq/freq_0
        #pwr = -2.5 + coeffs[1] + coeffs[2]*np.log(normfreq)
        fg = coeffs[0]*np.power(normfreq,-2.5) + coeffs[1]*np.power(normfreq, -2.5)*np.log(normfreq) + coeffs[2]*np.power(normfreq,-2.5)*np.power(np.log(normfreq), 2.0) + coeffs[3]*np.power(normfreq, -4.5) + coeffs[4]*np.power(normfreq, -2.0)
        #fg = coeffs[0]*np.power(normfreq, pwr)*np.exp(-coeffs[3]*np.power(normfreq,-2)) + coeffs[4]*np.power(normfreq,-2)
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-4]
        x0 = theta[-3]
        width = theta[-2]
        tau = theta[-1]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

# =============================================================================

class hills_physical(bmc.model):
    """
    Model used by Hills in 2019 paper eq.6

    Requires parameters in form
    theta = [b0, b1, b2, b3, Te, amp, x0, width, tau]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "physical polynomial + ionosphere"
        self.name_sig = "flat gaussian"
        self.labels = ["b0","b1","b2","b3","Te","amp","x0","width","tau"]
        pass

    def foreground(self, theta):
        """
        Physical foreground model
        """
        freq_0 = 75.0
        coeffs = theta[0:-4]
        normfreq = self.freq/freq_0
        pwr = -2.5 + coeffs[1] + coeffs[2]*np.log(normfreq)
        fg = coeffs[0]*np.power(normfreq, pwr)*np.exp(-coeffs[3]*np.power(normfreq,-2.0)) + coeffs[4]*(1 - np.exp(-coeffs[3]*np.power(normfreq, -2.0)))
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-4]
        x0 = theta[-3]
        width = theta[-2]
        tau = theta[-1]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

# =============================================================================

class hills_linear(bmc.model):
    """
    Model used by Hills in 2019 paper eq.6

    Requires parameters in form
    theta = [b0, b1, b2, b3, b4, amp, x0, width, tau]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "physical polynomial + ionosphere"
        self.name_sig = "flat gaussian"
        self.labels = ["b0","b1","b2","b3","b4","amp","x0","width","tau"]
        pass

    def foreground(self, theta):
        """
        Linearised physical foreground model
        """
        freq_0 = 75.0
        coeffs = theta[0:-4]
        normfreq = self.freq/freq_0
        pwr = -2.5 + coeffs[1] + coeffs[2]*np.log(normfreq)
        fg = coeffs[0]*np.power(normfreq, pwr)*np.exp(-coeffs[3]*np.power(normfreq,-2.0)) + coeffs[4]*(np.power(normfreq, -2.0))
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-4]
        x0 = theta[-3]
        width = theta[-2]
        tau = theta[-1]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

# =============================================================================

class hills_sine(bmc.model):
    """
    Model used by Hills in 2018

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, a5, amp, phi, l]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial_5th"
        self.name_sig = "sine function"
        self.labels = ["a0","a1","a2","a3","a4","a5","amp","phi","l"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 5th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-3]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def signal(self, theta):
        """
        Sine function
        """
        amp = theta[-3]
        phi = theta[-2]
        l = theta[-1]

        t21 =  amp * np.sin(2*np.pi*self.freq/l + phi)
        return t21

# ===============================================================================

class hillsphysical(bmc.model):
    """
    Model used by Hills in 2018 paper eq.6

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, amp, x0, width, tau]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "physical polynomial"
        self.name_sig = "flat gaussian"
        self.labels = ["a0","a1","a2","a3","a4","amp","x0","width","tau"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 4th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-4]
        normfreq = self.freq/freq_0
        pwr = -2.5 + coeffs[1] + coeffs[2]*np.log(normfreq)
        fg = coeffs[0]*np.power(normfreq, pwr)*np.exp(-coeffs[3]*np.power(normfreq,-2)) + coeffs[4]*np.power(normfreq,-2)
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-4]
        x0 = theta[-3]
        width = theta[-2]
        tau = theta[-1]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

# =============================================================================

class freenoise_bowman(bmc.model):
    """
    Model used by Bowman in 2018 paper eq.2, with noise as free parameter

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, amp, x0, width, tau, noise]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial"
        self.name_sig = "flat gaussian"
        self.labels = ["a0","a1","a2","a3","a4","amp","x0","width","tau","noise"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 4th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-5]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-5]
        x0 = theta[-4]
        width = theta[-3]
        tau = theta[-2]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise

# add more models here
