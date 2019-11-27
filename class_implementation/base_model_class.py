# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:46:13 2019

@author: matth
"""

# Creating a class structure for different foreground and signal models

# Importing modules
import numpy as np

# Base Model class
class model:
    """
    Base class for a foreground and signal model
    """
    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "Base"
        self.name_sig = "Base"
        self.labels = []  #list of names of parameters
        pass
    
    def __repr__(self):
        return "(FG:{self.name_fg} + SIG:{self.name_sig} nu=[{np.min(self.freq)}, {np.max(self.freq)}])"

    def __str__(self):
        return "(FG:{self.name_fg} + SIG:{self.name_sig} nu=[{np.min(self.freq)}, {np.max(self.freq)}])"

    def observation(self, theta, withFG = True, withSIG = True):
        """
        Return the full modelled observation
        """

        sig = 0.0
        fg =0.0

        if withSIG:
            sig = self.signal(theta)

        if withFG:
            fg = self.foreground(theta)

        return sig + fg

    def foreground(self, theta):
        """ Calculate foreground model
        """
        pass

    def signal(self, theta):
        """
        Calculate 21 cm signal model
        """
        pass
