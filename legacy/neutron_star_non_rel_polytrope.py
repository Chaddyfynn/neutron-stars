# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:27:59 2023

NEUTRON STAR with NON-RELATIVISTIC POLYTROPIC APPROXIMATION

@author: chadd
"""

import numpy as np
import scipy.constants as con
import calculator as solve

# RK4 / Calculator Settings
R_0 = 0.0001  # Initial Condition Radius, m
R_F = 25_000  # Final Radius, m
NUM = 1_000_000  # Number of Steps
STEP = R_F / NUM  # Step, dx


# System Settings
# STATE_0 = np.array([0, 1e32])  # Initial State at R_0, [kg, Pa] uni rk4
STATE_0 = [0, 1e32]  # Initial State at R_0, [kg, Pa] scipy
MIN_PRESSURE = 9e29  # Minimum Central Pressure, Pa
MAX_PRESSURE = 2e32  # Maximum Central Pressure, Pa
NUM_STEPS = 100  # Number of Iterations (Plot Points on Graph)
PRESSURE_STEP = (MAX_PRESSURE - MIN_PRESSURE) / NUM_STEPS
TOLERANCE = 0.00001  # 0 - 1 factor of max dp/dr root point tolerance (ideal=0)

# Astronomical Constant
M0 = 1.98847e30  # Solar Mass, kg
R0 = (con.G*M0)/(con.c**2)  # Solar Schwarzchild Radius, m

# System Constants
K = con.hbar**2/(15*np.pi**2*con.m_n) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(5/3)  # Pressure Constant, No Unis
GAMMA = 5/3  # Polytropic Index, No Units

# Save and Graph Settings
FILENAME = "Neutron_Star_Non_Rel_Polytrope"  # Graph and Text File Desired Name
PLOT_TIME = False  # Plot Function Evaluation Times vs Pressure? (Boolean)
# Compute for a range of central pressures (True), or one (False)
FULL_COMPUTATION = True
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
        # radii, states = solve.rk4(grad, R_0, STATE_0, STEP, NUM)
        radii, states = solve.scipy(grad, R_0, STATE_0, STEP, NUM)
        solve.plot(radii/1000, states, FILENAME)
        solve.save(radii/1000, states, FILENAME, METADATA)
    return None


def energy_density(radius, state):
    m, p = state
    return np.power(p / K, 3/5)


def grad(radius, state):
    m, p = state
    r_2 = np.power(radius, 2)
    dm_dr = ((4 * np.pi * r_2) * energy_density(radius, state)/(M0*(con.c)**2))
    dp_dr = -1 * (R0*m/r_2) * energy_density(radius, state)

    return np.array([dm_dr, dp_dr])  # gradient array


if __name__ == "__main__":
    main()
