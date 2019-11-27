# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:55:54 2019

@author: matth
"""

# importing modules
from __future__ import absolute_import, unicode_literals, print_function
from pymultinest.solve import solve
import os
try: os.mkdir('chains')
except OSError: pass


class multinest_object():
    """
    A class to run multinest sampling for a given dataset, model, and priors
    """
    def __init__(self, data, model, priors, loglike, output_prefix = "testoutput2-"): # might want to move loglike elsewhere
        self.data = data
        self.model = model
        self.priors = priors
        self.loglike = loglike
        self.prefix = output_prefix
    
    def solve_multinest(self):
        """
        Run pymultinest.solve, saving chain and parameter values
        """
        parameters = self.model.labels
        n_params = len(parameters)
        result = solve(LogLikelihood=self.loglike, Prior=self.priors, n_dims=n_params, outputfiles_basename=self.prefix, verbose=True)
        print()
        print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
        print()
        print('parameter values:')
        for name, col in zip(parameters, result['samples'].transpose()):
        	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
            
        # make marginal plots by running:
        # $ python multinest_marginals.py chains/3-
        # For that, we need to store the parameter names:
        import json
        with open('%sparams.json' % self.prefix, 'w') as f:
            json.dump(parameters, f, indent=2)
        
