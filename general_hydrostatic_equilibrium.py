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

# Imported Constants
C = con.c  # Speed of Light, m s^-1
G = con.G  # Gravitational Constant, kg^-2 m^2

# Modified Constants
C_2 = C**2  # Speed of Light Squared, m^2 s^-2

# RK4 Settings and Initial Conditions
R_0 = 1  # Radius for Initial State, m
STATE_0 = np.array([0, 22e21])  # State at R_), [Mass, Pressure], [kg, Pa]
STEP = 0.01  # RK4 Step Size
NUM = 120_000  # RK4 Steps

# RK4 Autosave Settings
FILENAME = "Hydrostatic"

# Thermodynamic Constants
K = 1.3e-29  # Pressure Constant
GAMMA = 5/3  # Polytropic Index


def main():
    radii, states = solve.rk4(polytropic_hydrostatic, R_0, STATE_0, STEP, NUM)
    mass = states[:, 0] / 1.989e30
    pressure = states[:, 1]
    graph_states = np.c_[mass, pressure]
    solve.plot(radii/1000, graph_states)
    if solve.save(radii, states, FILENAME) == 0:
        print("Save Successful")


def epsilon(radius, state):
    mass, pressure = state
    return np.power(pressure / K, 1/GAMMA)


def hydrostatic(radius, state):
    mass, pressure = state  # Mass and pressure at current radius
    # Radius Squared (saves multiple calcs later)
    radius_2 = np.power(radius, 2)

    dm_dr = 4 * np.pi * radius_2 * epsilon(radius, state) / C_2  # Mass Eqn
    dp_dr = -1 * G * epsilon(radius, state) * mass / \
        (C_2 * radius_2)  # Pressure Eqn

    return np.array([dm_dr, dp_dr])  # gradient array


def polytropic_hydrostatic(radius, state):
    mass, pressure = state  # Mass and pressure at current radius
    # Radius Squared (saves multiple calcs later)
    radius_2 = np.power(radius, 2)

    dm_dr = 4 * np.pi * radius_2 * epsilon(radius, state) / C_2  # Mass Eqn
    dp_dr = -1 * G * epsilon(radius, state) * mass / \
        (C_2 * radius_2)  # Pressure Eqn

    return np.array([dm_dr, dp_dr])  # gradient array


if __name__ == "__main__":
    main()
