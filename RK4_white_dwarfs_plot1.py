# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:43:22 2023

@author: Roryt
"""

import RK4_built_in as solve
import matplotlib.pyplot as plt

radius, state = solve.main()


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
    plt.savefig("WD_plot1_rel.png", dpi=1000)
    plt.show()
    plt.clf()
    return 0


plot(radius, state)
