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
MAX_RADIUS = 20_000_000  # m
NUM = 20000 * MULTIPLIER  # Number of Steps
STEP = MAX_RADIUS / NUM  # Step, dx


# System Settings
STATE_0 = np.array([0, 5.62e24])
MIN_PRESSURE = 1  # Minimum Central Pressure, Pa
MAX_PRESSURE = 5e24  # Maximum Central Pressure, Pa
NUM_STEPS = 100  # Number of Iterations (Plot Points on Graph)
PRESSURE_STEP = (MAX_PRESSURE - MIN_PRESSURE) / NUM_STEPS
TOLERANCE = 0.002  # dp/dr root point tolerance (ideal=0)

# Astronomical Constant
M0 = 1.98847e30  # Solar Mass, kg
R0 = (con.G*M0)/(con.c**2)  # Solar Schwarzchild Radius, m

# System Constants
K = con.hbar * con.c / (12 * np.pi**2) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(4/3)  # Pressure Constant, No Unis
GAMMA = 4/3  # Polytropic Index, No Units

# Save and Graph Settings
FILENAME = "WD_REL_REPEAT"  # Graph and Text File Desired Name
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
        new_states = [[],[]]
        for i in range(len(states)):
            new_states[0].append(states[i][0])
            new_states[1].append(states[i][1])
        new_states = np.array(new_states)
        radii = radii/1000
        radius, mass = solve.root_prev(radii, new_states, TOLERANCE)
        solve.plot_root(radii, new_states, FILENAME, radius)
        # solve.save(radii, new_states, FILENAME, METADATA)
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
