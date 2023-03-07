# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:27:59 2023

@author: chadd
"""

import numpy as np
import scipy.constants as con
import calculator as solve

# RK4 / Calculator Settings
MULTIPLIER = 2  # Step and Number Multiplier (Higher =>  More Resolution)
R_0 = 0.001  # Initial Condition Radius, m
STEP = 1000 / MULTIPLIER  # Step, dx
NUM = 17000 * MULTIPLIER  # Number of Steps

# System Settings
MIN_PRESSURE = 1  # Minimum Central Pressure, Pa
MAX_PRESSURE = 10e21  # Maximum Central Pressure, Pa
NUM_STEPS = 100  # Number of Iterations (Plot Points on Graph)
PRESSURE_STEP = (MAX_PRESSURE - MIN_PRESSURE) / NUM_STEPS

# Astronomical Constant
M0 = 1.98847e30  # Solar Mass, kg
R0 = (con.G*M0)/(con.c**2)  # Solar Schwarzchild Radius, m

# System Constants
K = con.hbar * con.c / (12 * np.pi**2) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(4/3)  # Pressure Constant, No Unis
GAMMA = 4/3  # Polytropic Index, No Units

# Save and Graph Settings
FILENAME = "White_Dwarf_Rel_Polytrope"  # Graph and Text File Desired Name
PLOT_TIME = False  # Plot Function Evaluation Times vs Pressure? (Boolean)
METADATA = [R_0, STEP, NUM, MIN_PRESSURE, MAX_PRESSURE, NUM_STEPS, K, GAMMA]


def main():
    pressures, radii, masses = solve.iterate(
        grad, R_0, STEP, NUM, MIN_PRESSURE, MAX_PRESSURE, PRESSURE_STEP, FILENAME, PLOT_TIME)
    solve.plot_pressure(pressures, radii, masses, FILENAME)
    states = np.c_[radii, masses]
    solve.save(pressures, states, FILENAME, METADATA)
    return None


def grad(radius, state):
    m, p = state
    dm_dr = ((4 * np.pi * np.power((radius), 2) *
             np.power((p/K), 1/GAMMA))/(M0*(con.c)**2))
    dp_dr = -1 * (R0*m/(np.power((radius), 2))) * \
        np.power((p/K), (1/GAMMA))

    return np.array([dm_dr, dp_dr])  # gradient array


if __name__ == "__main__":
    main()
