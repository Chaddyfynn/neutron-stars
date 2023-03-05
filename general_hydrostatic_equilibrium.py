# -*- coding: utf-8 -*-
"""
|/--------------------TITLE--------------------\|
PHYS20161 -- Assignment N -- [ASSIGNMENT NAME]
|/---------------------------------------------\|

[CODE DESCRIPTION/USAGE/OUTLINE]


Created: Thu Mar  2 11:00:33 2023
Last Updated:

@author: Charlie Fynn Perkins, UID: 10839865 0
"""
import RK4 as solve
import scipy.constants as con
import numpy as np
import ideal_filename as check
from scipy.integrate import solve_ivp

"""
The convention in this document for units are as follows:
    Mass: kg
    Space: m
    Time: s
    Pressure: Pa

    SI UNITS ONLY
"""

# Imported Constants
C = con.c  # Speed of Light, m s^-1
G = con.G  # Gravitational Constant, kg^-2 m^2

# Modified Constants
C_2 = C**2  # Speed of Light Squared, m^2 s^-2

# RK4 Settings and Initial Conditions (UoM)
R_0 = 1  # Radius for Initial State, m
STATE_0 = np.array([0, 2.2e21])  # State at R_0), [Mass, Pressure], [kg, Pa]
STEP = 1  # RK4 Step Size
NUM = 10_000  # RK4 Steps OR End Radius (RK4 Step Till)

# Solve_IVP Settings
INTERVAL = (0, NUM)
SPAN = np.linspace(0, NUM, 100)

# RK4 Autosave Settings
FILENAME = "Hydrostatic"

# Thermodynamic Constants
K = 1.7e-22  # Pressure Constant, no units
GAMMA = 5/3  # Polytropic Index, no units

# Scaling/Unit Constants
M_SUN = 1.989e30  # Mass of Sun, kg
R_SCHW = G * M_SUN / C_2  # Solar Schwarzchild Radius, m


def main():
    # radii, states = solve.rk4_step_till(
    #     paper, R_0, STATE_0, STEP, NUM)
    output = solve_ivp(polytropic_hydrostatic, INTERVAL, STATE_0, t_eval=SPAN)
    filename = check.filename("scipy_output", ".txt")
    np.savetxt(filename, output, delimiter=",")
    # mass = states[:, 0]
    # pressure = states[:, 1]
    # graph_states = np.c_[mass, pressure]
    # solve.plot(radii, graph_states)
    # if solve.save(radii, states, FILENAME) == 0:
    #     print("Save Successful")


def epsilon(radius, state):
    mass, pressure = state
    return np.power(pressure / K, 1/GAMMA)


def polytropic_hydrostatic(radius, state):
    mass, pressure = state  # Mass and pressure at current radius
    # Radius Squared (saves multiple calcs later)
    radius_2 = np.power(radius, 2)

    dm_dr = 4 * np.pi * radius_2 * epsilon(radius, state) / C_2  # Mass Eqn
    dp_dr = -1 * G * epsilon(radius, state) * mass / \
        (C_2 * radius_2)  # Pressure Eqn

    return np.array([dm_dr, dp_dr])  # gradient array


def paper(radius, state):
    mass, pressure = state  # Mass and pressure at current radius
    # Radius Squared (saves multiple calcs later)
    radius_2 = np.power(radius, 2)

    dm_dr = 4 * np.pi * radius_2 * \
        epsilon(radius, state) / (C_2 * M_SUN)  # Mass Eqn
    dp_dr = -1 * R_SCHW * epsilon(radius, state) * mass / \
        (radius_2)  # Pressure Eqn

    return np.array([dm_dr, dp_dr])  # gradient array


if __name__ == "__main__":
    main()
