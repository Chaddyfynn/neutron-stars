# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:27:59 2023

NEUTRON STAR with ULTRARELATIVISTIC POLYTROPIC APPROXIMATION

@author: chadd
"""

import numpy as np
import scipy.constants as con
import calculator as solve

# RK4 / Calculator Settings
MULTIPLIER = 1  # Step and Number Multiplier (Higher =>  More Resolution)
R_0 = 0.001  # Initial Condition Radius, m
STEP = 1 / MULTIPLIER  # Step, dx
NUM = 50000 * MULTIPLIER  # Number of Steps

# System Settings
STATE_0 = np.array([0, 1e32])
MIN_PRESSURE = 1  # Minimum Central Pressure, Pa
MAX_PRESSURE = 1e33  # Maximum Central Pressure, Pa
NUM_STEPS = 100  # Number of Iterations (Plot Points on Graph)
PRESSURE_STEP = (MAX_PRESSURE - MIN_PRESSURE) / NUM_STEPS
TOLERANCE = 0.002  # dp/dr root point tolerance (ideal=0)

# Astronomical Constant
M0 = 1.98847e30  # Solar Mass, kg
R0 = (con.G*M0)/(con.c**2)  # Solar Schwarzchild Radius, m

# System Constants
K = 1  # Pressure Constant, No Unis
GAMMA = 1  # Polytropic Index, No Units

# Save and Graph Settings
FILENAME = "White_Dwarf_Rel_Polytrope"  # Graph and Text File Desired Name
PLOT_TIME = False  # Plot Function Evaluation Times vs Pressure? (Boolean)
# Compute for a range of central pressures (True), or one (False)
FULL_COMPUTATION = False
PLOT_INDIVIDUAL = False  # Create graphs for each central pressure (False)
CROP = 0  # Left Crop for Full computation, 5e23 for rel
METADATA = [R_0, STEP, NUM, MIN_PRESSURE, MAX_PRESSURE, NUM_STEPS, K, GAMMA]


def main():
    if FULL_COMPUTATION:
        pressures, radii, masses = solve.iterate(
            grad, R_0, STEP, NUM, MIN_PRESSURE, MAX_PRESSURE, PRESSURE_STEP, TOLERANCE, FILENAME, PLOT_TIME, PLOT_INDIVIDUAL)
        solve.plot_pressure(pressures, radii, masses, FILENAME, CROP)
        states = np.c_[radii, masses]
        solve.save(pressures, states, FILENAME, METADATA)
    else:
        radii, states = solve.rk4(grad, R_0, STATE_0, STEP, NUM)
        solve.plot(radii, states, FILENAME)
        solve.save(radii, states, FILENAME, METADATA)
    return None


def energy_density(radius, state):
    m, p = state
    return 3 * p


def grad(radius, state):
    m, p = state
    r_2 = np.power(radius, 2)
    dm_dr = ((4 * np.pi * r_2) * energy_density(radius, state)/(M0*(con.c)**2))
    dp_dr = -1 * (R0*m/r_2) * energy_density(radius, state)

    return np.array([dm_dr, dp_dr])  # gradient array


if __name__ == "__main__":
    main()
