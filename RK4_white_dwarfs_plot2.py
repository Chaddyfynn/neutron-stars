# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:50:15 2023

@author: Roryt
"""

import RK4_p0_function as solve
import numpy as np
import matplotlib.pyplot as plt


p_max = 1e20
p = 1e25
p0 = []
r_vals = []
m_vals = []

rval, mval = solve.main(p)


for i in range(int(1e23), int(5e25), int(2.5e24)):
    rval, mval = solve.main(i)
    p0.append(i*10)
    r_vals.append(rval/1000)
    m_vals.append(mval)


def plot():
    # Prepare two side by side plots
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    ax2 = ax1.twinx()

    # Axis 1: Show the different state variables against time
    ax1.set(xlabel="Pressure, dyne/cm^2")
    ax2.set(ylabel="Mass, Solar Masses")
    ax1.set(ylabel="Radius, km")
    ax1.plot(p0, r_vals, color="red", label="Radius")
    ax2.plot(p0, m_vals, linestyle="--",
             color="blue", label="Mass")
    ax2.set_ylim([1.4, 1.44])
    ax1.legend()
    ax2.legend()

    # Show and close the plot
    ax1.grid()
    # ax3.grid()
    plt.tight_layout()
    plt.savefig("Mass_Radius_rel_pol_WD.png", dpi=1000)
    plt.show()
    plt.clf()
    return 0


plot()
