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


def rk4_step(grad, time, state, step_size):
    # Calculate various midpoint k states
    k1 = grad(time, state)*step_size
    k2 = grad(time+step_size/2, state+k1/2)*step_size
    k3 = grad(time+step_size/2, state+k2/2)*step_size
    k4 = grad(time+step_size, state+k3)*step_size
    # Return new time and state
    return time+step_size, state+(k1/2 + k2 + k3 + k4/2)/3

# Function for taking n steps using RK4


def rk4_n_steps(grad, time, state, step_size, n_steps):
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
    dm_dr = ((4 * np.pi * np.power(radius, 2)) /
             (M0*c**2)) * np.power((p/K), 1/GAMMA)
    dp_dr = -1 * (R0/(np.power(radius, 2))) * \
        (p/K)**(1/GAMMA) * m

    '''
    m, p = state
    dm_dr = ((4 * np.pi * np.power(radius, 2)) /
             (M0*c**2)) * (p/K)**(1/GAMMA)
    dp_dr = -1 * (R0/(np.power(radius, 2))) * \
        (p/K)**(1/GAMMA) * m
    '''
    return np.array([dm_dr, dp_dr])  # gradient array


# Initial state is [m, p] at time = 0
t_0 = 1
state_0 = np.array([0, 2.2e22])  # [Mass, Pressure]

c = 3e8
M0 = 1.989e30
R0 = (con.G*M0)/(c**2) * 0.001

K = 1e-29
GAMMA = 5/3

times, states = rk4_n_steps(grad, t_0, state_0, 1, 12_700)
print(times)
print(states)

# Prepare two side by side plots
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
ax2 = ax1.twinx()

# Axis 1: Show the different state variables against time
ax1.set(xlabel="Radius r, km")
ax2.set(ylabel="Mass, Solar Masses")
ax1.set(ylabel="Pressure, dyne/cm^2")
ax2.plot(times, states[:, 0], color="red", label="Mass")
ax1.plot(times, states[:, 1], linestyle="--", color="blue", label="Pressure")
ax1.legend()
ax2.legend()

# Axis 2: Show the x,y plane
#ax3.set(xlabel="Mass", ylabel="Pressure")
#ax3.plot(states[:, 0], states[:, 1], label="Mass 1")


# Show and close the plot
ax1.grid()
# ax3.grid()
plt.tight_layout()
plt.show()
plt.clf()
