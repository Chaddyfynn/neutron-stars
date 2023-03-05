# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:27:59 2023

@author: chadd
"""

import numpy as np
import scipy.constants as con
import RK4 as solve

R_0 = 0.001
STATE_0 = np.array([0, 2.2e22/10])  # [Mass, Pressure]
STEP = 1000
NUM = 13000

MIN_PRESSURE = 1
MAX_PRESSURE = 5e21
NUM_STEPS = 100
PRESSURE_STEP = (MAX_PRESSURE - MIN_PRESSURE) / NUM_STEPS


M0 = 1.98847e30

R0 = (con.G*M0)/(con.c**2)

K = con.hbar**2/(15*np.pi**2*con.m_e) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(5/3)

GAMMA = 5/3

FILENAME = "White_Dwarf"


def main():
    pressures, radii, masses = solve.iterate(grad, R_0, STEP, NUM, MIN_PRESSURE, MAX_PRESSURE, PRESSURE_STEP, FILENAME)
    solve.plot_pressure(pressures, radii, masses, FILENAME)
    states = np.c_[radii, masses]
    solve.save(pressures, states, FILENAME)
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
