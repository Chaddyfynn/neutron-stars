# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:42:04 2023

@author: Roryt
"""

import numpy as np
import scipy.constants as con
import scipy.integrate as sci_int
import matplotlib.pyplot as plt


r_0 = 0.001
r_max = 12000
t_span = [0.001, 12000000]
state_0 = np.array([0, 2.2e22/10])  # [Mass, Pressure]


M0 = 1.98847e30

R0 = (con.G*M0)/(con.c**2)

K = con.hbar**2/(15*np.pi**2*con.m_e) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(5/3)

GAMMA = 5/3


def main():
    solution = sci_int.solve_ivp(grad, t_span, state_0, method='RK45',
                                 t_eval=None, dense_output=False, events=None,
                                 vectorized=True, args=None, max_step=1000)

    radius = solution['t']
    state = solution['y']

    plot(radius, state)


def grad(radius, state):
    m, p = state
    dm_dr = ((4 * np.pi * np.power((radius), 2) *
             np.power((p/K), 1/GAMMA))/(M0*(con.c)**2))
    dp_dr = -1 * (R0*m/(np.power((radius), 2))) * \
        np.power((p/K), (1/GAMMA))

    return np.array([dm_dr, dp_dr])


def plot(radii, states):
    # Prepare two side by side plots
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ax2 = ax1.twinx()

    # Axis 1: Show the different state variables against time
    ax1.set(xlabel="Radius r, km")
    ax2.set(ylabel="Mass, Solar Masses")
    ax1.set(ylabel="Pressure, dyne/cm^2")
    ax2.plot(radii/1000, states[0], color="red", label="Mass")
    ax1.plot(radii/1000, states[1]*10, linestyle="--",
             color="blue", label="Pressure")
    ax1.legend()
    ax2.legend()

    # Show and close the plot
    ax1.grid()
    # ax3.grid()
    plt.tight_layout()
    plt.savefig("Figure.png", dpi=1000)
    plt.show()
    plt.clf()
    return 0


main()
