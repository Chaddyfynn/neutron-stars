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


c = 1
k = 1
w = k * c


def grad(time, state):

    E = state[0]
    E_dot = state[1]

    E_dot_dot = -w**2 * E

    return np.array([E_dot, E_dot_dot])  # gradient array


# Initial state is [x_1, x_2, v_1, v_2] at time = 0
t_0 = 0
state_0 = np.array([1, 1])  # [E, E_dot]

times, states = rk4_n_steps(grad, t_0, state_0, 0.01, 1_000)

# Prepare two side by side plots
fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))

# Axis 1: Show the different state variables against time
ax1.set(xlabel="t")
ax1.plot(times, states[:, 0], label="x_1")
ax1.plot(times, states[:, 1], linestyle="--", label="x_2")
ax1.legend()

# Axis 2: Show the x,y plane
ax2.set(xlabel="x", ylabel="v")
ax2.plot(states[:, 0], states[:, 1], label="Mass 1")


# Show and close the plot
plt.show()
plt.clf()
