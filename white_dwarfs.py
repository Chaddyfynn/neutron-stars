# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:27:59 2023

@author: chadd
"""

import numpy as np
import scipy.constants as con
import RK4 as solve

R_0 = 1 # Start Radius
STATE_0 = np.array([0, 2.2e22])  # Boundary Conditions at R=R_0 [Mass, Pressure]
STEP = 1 # Step Size
NUM = 12_700 # Number of Steps
 
c = con.c # Speed of Light, m/s
M_SUN = 1.989e30 # Mass of sun in kg
R_SCHW = (con.G*M_SUN)/(c**2) # Solar Schwarzschild Radius

K = 1e-29 # Pressure Constant
GAMMA = 5/3 

def main():
    radii, states = solve.rk4(grad, R_0, STATE_0, STEP, NUM)
    solve.plot(radii, states)
    
def grad(radius, state):
    m, p = state
    dm_dr = ((4 * np.pi * np.power(radius, 2)) /
             (M_SUN*c**2)) * np.power((p/K), 1/GAMMA)
    dp_dr = -1 * (R_SCHW/(np.power(radius, 2))) * \
        (p/K)**(1/GAMMA) * m
        
    return np.array([dm_dr, dp_dr])  # gradient array

if __name__ == "__main__":
   main()