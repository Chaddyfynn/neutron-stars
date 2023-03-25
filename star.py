# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:04:59 2023

@author: chadd
"""

class Star():  # Star Superclass
    def __init__(self, central_pressure):
        super(Star, self).__init__()
        self.p0 = central_pressure  # Create p0 attribute

    def __str__(self):  # String formatting
        return f"Generic Star: p0={self.p0} Pa"

    def increment(self, step):  # Central Pressure Increment Function
        self.p0 += step  # Increase p0 by step
        return None