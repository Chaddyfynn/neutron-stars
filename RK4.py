# -*- coding: utf-8 -*-
"""
|/-----------------RK4 COMPUTER----------------\|

|/---------------------------------------------\|


Created: Fri Feb  3 18:25:36 2023
Last Updated:

@author: Department of Physics and Astronomy,
         University of Manchester
"""
import numpy as np
import scipy.constants as con
import matplotlib.pyplot as plt

# Function for taking a single RK4 step
# The grad function must be of form grad(time (float), state (numpy array of floats)) -> (numpy array of floats)

# Initial state is [m, p] at time = 0
r_0 = 0.001
state_0 = np.array([0, 2.2e22/10])  # [Mass, Pressure]


M0 = 1.98847e30

R0 = (con.G*M0)/(con.c**2)

K = con.hbar**2/(15*np.pi**2*con.m_e) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(5/3)

GAMMA = 5/3


def main():
    radii, states = rk4(grad, r_0, state_0, 1000, 13000)
    if plot(radii, states) == 0:
        print("Success")
        d = np.where(states[:, 1] == min(states[:, 1]))
        index = d[0][0]
        print('White dwarf radius is', radii[index]/1000, 'km.')
        print('White dwarf mass is', states[index, 0])
    else:
        print("Failure")


def rk4_step(grad, time, state, step_size):
    # Calculate various midpoint k states
    k1 = grad(time, state)*step_size
    k2 = grad(time+step_size/2, state+k1/2)*step_size
    k3 = grad(time+step_size/2, state+k2/2)*step_size
    k4 = grad(time+step_size, state+k3)*step_size
    # Return new time and state
    return time+step_size, state+(k1/2 + k2 + k3 + k4/2)/3

# Function for taking n steps using RK4


def rk4(grad, time, state, step_size, n_steps):
    # Prepare numpy arrays for storing data
    times = np.array([time, ])
    state_arr = np.empty(shape=(0, state.size))
    # We will use vstack to add new time slices the state array
    state_arr = np.vstack((state_arr, state))

    # Take n steps
    for _ in range(n_steps):
        new_time, new_state = rk4_step(
            grad, times[-1], state_arr[-1], step_size)
        times = np.append(times, new_time)
        state_arr = np.vstack((state_arr, new_state))

    return times, state_arr

# Function for taking steps till some final time using RK4. Useful for comparing step sizes


def rk4_step_till(grad, time, state, step_size, final_time):
    # Prepare numpy arrays for storing data
    times = np.array([time, ])
    state_arr = np.empty(shape=(0, state.size))
    # We will use vstack to add new time slices the state array
    state_arr = np.vstack((state_arr, state))

    # Take as many steps as needed
    while times[-1] < final_time:
        new_time, new_state = rk4_step(
            grad, times[-1], state_arr[-1], step_size)
        times = np.append(times, new_time)
        state_arr = np.vstack((state_arr, new_state))

    return times, state_arr


def grad(radius, state):
    m, p = state
    dm_dr = ((4 * np.pi * np.power((radius), 2) *
             np.power((p/K), 1/GAMMA))/(M0*(con.c)**2))
    dp_dr = -1 * (R0*m/(np.power((radius), 2))) * \
        np.power((p/K), (1/GAMMA))

    return np.array([dm_dr, dp_dr])  # gradient array


def plot(radii, states):
    # Prepare two side by side plots
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ax2 = ax1.twinx()

    # Axis 1: Show the different state variables against time
    ax1.set(xlabel="Radius r, km")
    ax2.set(ylabel="Mass, Solar Masses")
    ax1.set(ylabel="Pressure, dyne/cm^2")
    ax2.plot(radii/1000, states[:, 0], color="red", label="Mass")
    ax1.plot(radii/1000, states[:, 1]*10, linestyle="--",
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


if __name__ == "__main__":
    main()
