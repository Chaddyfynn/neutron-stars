# -*- coding: utf-8 -*-
"""
|/-----------------RK4 COMPUTER----------------\|

|/---------------------------------------------\|


Created: Fri Feb  3 18:25:36 2023
Last Updated:

@author: Department of Physics and Astronomy,
         University of Manchester
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time as tm

# Function for taking a single RK4 step
# The grad function must be of form grad(time (float), state (numpy array of floats)) -> (numpy array of floats)


def main():
    print("This file is a collection of functions, not a runnable file")


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


def plot(radii, states, ideal_filename):
    # Prepare two side by side plots
    print("Plotting...")
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ax2 = ax1.twinx()

    # Axis 1: Show the different state variables against time
    ax1.set(xlabel="Radius r, km")
    ax2.set(ylabel="Mass, Solar Masses")
    ax1.set(ylabel="Pressure, Pa")
    ax2.plot(radii, states[:, 0], color="red", label="Mass")
    ax1.plot(radii, states[:, 1], linestyle="--",
             color="blue", label="Pressure")
    ax1.legend()
    ax2.legend()

    # Show and close the plot
    ax1.grid()
    # ax3.grid()
    plt.tight_layout()
    filename = path_checker(ideal_filename, ".png")
    print("Saving figure...")
    plt.savefig(filename, dpi=1000)
    plt.show()
    plt.clf()
    return 0


def plot_pressure(pressures, radii, masses, ideal_filename):
    # Prepare two side by side plots
    print("Plotting...")
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ax2 = ax1.twinx()

    # Axis 1: Show the different state variables against time
    ax1.set(xlabel="Pressure, Pa")
    ax2.set(ylabel="Mass, Solar Masses")
    ax1.set(ylabel="Radius, km")
    ax2.plot(pressures, masses, color="red", label="Mass")
    ax1.plot(pressures, radii, linestyle="--",
             color="blue", label="Radius")
    ax1.legend()
    ax2.legend()

    # Show and close the plot
    ax1.grid()
    # ax3.grid()
    plt.tight_layout()
    print("Saving figure...")
    filename = path_checker(ideal_filename, ".png")
    plt.savefig(filename, dpi=1000)
    plt.show()
    plt.clf()
    return 0


def plot_times(pressures, times):
    # Prepare two side by side plots
    print("Plotting...")
    filename = path_checker("Function_Time", ".png")
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    # Axis 1: Show the different state variables against time
    ax1.set(xlabel="Iteration")
    ax1.set(ylabel="Function Time, s")
    ax1.plot(pressures, times, '.', color="blue")
    # Show and close the plot
    ax1.grid()
    # ax3.grid()
    plt.tight_layout()
    print("Saving figure...")
    plt.savefig(filename, dpi=1000)
    plt.show()
    plt.clf()
    return 0


def path_checker(ideal_filename, extension):
    filename = ideal_filename
    address = "./saves/"  # Folder Address
    number = int(0)
    path = Path(address + filename + extension)  # Initial Path

    while path.is_file():
        number = number + int(1)
        number_append = str(number)
        filename = ideal_filename + '_' + number_append
        path = Path(address + filename + extension)
    return address + filename + extension


def save(radii, states, ideal_filename, metadata):
    print("Saving Array...")
    output_array = np.c_[radii, states]  # Output Array
    file = path_checker(ideal_filename, ".txt")
    output_meta = str("")
    for meta in metadata:
        output_meta = output_meta + ", " + str(meta)
    np.savetxt(file, output_array, delimiter=",",
               header=str(tm.ctime()), footer=output_meta)


def root(radii, states):
    index = np.where(states[:, 1] == min(states[:, 1]))[0][0]
    radius = radii[index]
    mass = states[index, 0]
    return radius, mass


def iterate(grad, r_0, step, num, min_pressure, max_pressure, pressure_step, filename, plot_time):
    whole_start = tm.time()
    pressure = min_pressure
    radii_output = np.zeros((0, 1))
    mass_output = np.zeros((0, 1))
    pressures = np.zeros((0, 1))
    times = np.zeros((0, 1))
    iterations = np.linspace(0, (max_pressure - min_pressure)/pressure_step, int((max_pressure - min_pressure)/pressure_step))
    while pressure < max_pressure:
        print("Calculating at pressure ", pressure, "Pa ...")
        start_time = tm.time()
        state_0 = np.array([0, pressure])
        pressures = np.append(pressures, pressure)
        radii, states = rk4(grad, r_0, state_0, step, num)
        radius, mass = root(radii/1000, states)
        radii_output = np.append(radii_output, radius)
        mass_output = np.append(mass_output, mass)
        pressure = pressure + pressure_step
        function_time = tm.time() - start_time
        times = np.append(times, function_time)
        print("Finished calculation in ", round(function_time, 1), "s ...")
    whole_time = tm.time() - whole_start
    print("Computation finished in ", round(whole_time, 1), "s")
    if plot_time:
        plot_times(iterations, times)

    return pressures, radii_output, mass_output


if __name__ == "__main__":
    main()
