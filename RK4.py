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


def eps(radius, state):
    return np.power(state[1]/K, 1/GAMMA)


def grad(radius, state):

    dm_dr = 4 * np.pi * np.power(radius, 2) * \
        eps(radius, state) / np.power(con.c, 2)
    dp_dr = -1 * con.G * eps(radius, state) * \
        state[0] / (np.power(con.c * radius, 2))
    # print(dp_dr)
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
state_0 = np.array([0, 2e22])  # [Mass, Pressure]

Z = 1
M_N = 1e-27
A = 1

c = 3e8
M0 = 1.989e30
R0 = (con.G*M0)/(c**2)


#K = np.power(con.hbar, 2) / (15 * np.power(np.pi, 2) *con.m_e) \
#     (3 * np.power(np.pi, 2) * Z / (M_N * np.power(con.c, 2) * A))
# print(K)
K = 6
GAMMA = 5/3

times, states = rk4_n_steps(grad, t_0, state_0, 10, 1200)
print(times)
print(states)

# Prepare two side by side plots
fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

# Axis 1: Show the different state variables against time
ax1.set(xlabel="Radius r, meters")
ax1.plot(times, states[:, 0], label="Mass")
ax1.plot(times, states[:, 1], linestyle="--", label="Pressure")
ax1.legend()

# Axis 2: Show the x,y plane
#ax2.set(xlabel="Radius r, meters", ylabel="Mass, kg")
#ax2.plot(states[:, 0], states[:, 1], label="Mass 1")


# Show and close the plot
plt.show()
plt.clf()
