# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:27:59 2023

@author: chadd
"""

import numpy as np
import scipy.constants as con
import calculator as solve

# RK4 / Calculator Settings
MULTIPLIER = 3  # Step and Number Multiplier (Higher =>  More Resolution)
R_0 = 0.001  # Initial Condition Radius, m
STEP = 1000 / MULTIPLIER  # Step, dx
NUM = 12000 * MULTIPLIER  # Number of Steps

# System Settings
STATE_0 = np.array([0, 2.2e21])
MIN_PRESSURE = 1  # Minimum Central Pressure, Pa
MAX_PRESSURE = 10e21  # Maximum Central Pressure, Pa
NUM_STEPS = 200  # Number of Iterations (Plot Points on Graph)
PRESSURE_STEP = (MAX_PRESSURE - MIN_PRESSURE) / NUM_STEPS

# Astronomical Constant
M0 = 1.98847e30  # Solar Mass, kg
R0 = (con.G*M0)/(con.c**2)  # Solar Schwarzchild Radius, m

# System Constants
K = con.hbar**2/(15*np.pi**2*con.m_e) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(5/3)  # Pressure Constant, No Unis
GAMMA = 5/3  # Polytropic Index, No Units

# Save and Graph Settings
FILENAME = "White_Dwarf_Non_Rel_Polytrope"  # Graph and Text File Desired Name
PLOT_TIME = True  # Plot Function Evaluation Times vs Pressure? (Boolean)
FULL_COMPUTATION = False
CROP = 0
METADATA = [R_0, STEP, NUM, MIN_PRESSURE, MAX_PRESSURE, NUM_STEPS, K, GAMMA]


def main():
    if FULL_COMPUTATION:
        pressures, radii, masses = solve.iterate(
            grad, R_0, STEP, NUM, MIN_PRESSURE, MAX_PRESSURE, PRESSURE_STEP, FILENAME, PLOT_TIME)
        solve.plot_pressure(pressures, radii, masses, FILENAME, CROP)
        states = np.c_[radii, masses]
        solve.save(pressures, states, FILENAME, METADATA)
    else:
        radii, states = solve.rk4(grad, R_0, STATE_0, STEP, NUM)
        solve.plot(radii, states, FILENAME)
        solve.save(radii, states, FILENAME, METADATA)
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
