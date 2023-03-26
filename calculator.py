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
import scipy.integrate as sci_int

M0 = 1.98847e30  # Solar Mass, kg

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


def scipy(grad, r_0, state_0, step, num):
    t_span = [r_0, step*num]
    solution = sci_int.solve_ivp(grad, t_span, state_0, method='RK45',
                                 t_eval=None, dense_output=False, events=None,
                                 vectorized=True, args=None, max_step=1000)

    radius = solution['t']
    state = solution['y']
    mass = state[0]
    pressure = state[1]
    state = mass, pressure
    return radius, state


def plot_root(radii, states, ideal_filename, radius):
    # mass = states[:, 0] # uni rk4
    # pressure = states[:, 1] # uni rk4
    mass, pressure = states  # scipy
    # Prepare two side by side plots
    if mass[-1] > 1e10:
        mass = mass / M0
    print("Plotting...")
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ax2 = ax1.twinx()

    # Axis 1: Show the different state variables against time
    ax1.set(xlabel="Radius r, km")
    ax2.set(ylabel="Mass, Solar Masses")
    ax1.set(ylabel="Pressure, Pa")
    ax2.plot(radii, mass, color="red", label="Mass")
    ax1.plot(radii, pressure, linestyle="--",
             color="blue", label="Pressure")
    ax1.axvline(radius, color='g')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
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


def plot(radii, states, ideal_filename):
    # mass = states[:, 0] # uni rk4
    # pressure = states[:, 1] # uni rk4
    mass, pressure = states
    # Prepare two side by side plots
    if mass[-1] > 1e10:
        mass = mass / M0
    print("Plotting...")
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ax2 = ax1.twinx()

    # Axis 1: Show the different state variables against time
    ax1.set(xlabel="Radius r, km")
    ax2.set(ylabel="Mass, Solar Masses")
    ax1.set(ylabel="Pressure, Pa")
    ax2.plot(radii, mass, color="red", label="Mass")
    ax1.plot(radii, pressure, linestyle="--",
             color="blue", label="Pressure")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
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


def plot_pressure(pressures, radii, masses, ideal_filename, crop):
    # Prepare two side by side plots
    print("Plotting...")
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ax2 = ax1.twinx()
    if masses[0] > 1e10:
        masses = masses / M0
    # Axis 1: Show the different state variables against time
    ax1.set(xlabel="Pressure, Pa")
    ax2.set(ylabel="Mass, Solar Masses")
    ax1.set(ylabel="Radius, km")
    ax2.plot(pressures, masses, color="red", label="Mass")
    ax1.plot(pressures, radii, linestyle="--",
             color="blue", label="Radius")
    ax1.set_xscale("log")
    ax2.set_xscale("log")
    # ax1.set_xlim(left=crop)
    # ax2.set_xlim(left=crop)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

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
    if isinstance(states, np.ndarray):
        if states.shape != radii.shape:
            masses = states[:, 0]
            pressures = states[:, 1]
            intermediate_array = np.c_[radii, masses]  # Output Array
            output_array = np.c_[intermediate_array, pressures]
        else:
            output_array = np.c_[radii, states]
    elif isinstance(states, tuple):
        if len(states) != len(radii):
            masses, pressures = states
            intermediate_array = np.c_[radii, masses]  # Output Array
            output_array = np.c_[intermediate_array, pressures]
    else:
        print("Array type undetermined")
        return None

    file = path_checker(ideal_filename, ".txt")
    output_meta = str("")
    for meta in metadata:
        output_meta = output_meta + ", " + str(meta)
    np.savetxt(file, output_array, delimiter=",",
               header=str(tm.ctime()), footer=output_meta)
    print("Array Saved ...")


def root_next(radii, states, tolerance):
    # masses = states[:, 0] # uni rk4
    # pressures = states[:, 1] # uni rk4
    masses, pressures = states  # scipy
    counter = 0
    while counter < len(radii) - 2:
        pressure_2 = pressures[counter + 2]
        pressure_1 = pressures[counter + 1]
        pressure_0 = pressures[counter]
        radius_2 = radii[counter + 2]
        radius_1 = radii[counter + 1]
        radius_0 = radii[counter]

        dp_0 = pressure_1 - pressure_0
        dp_1 = pressure_2 - pressure_1
        dr_0 = radius_1 - radius_0
        dr_1 = radius_2 - radius_1
        d2_p = dp_1 - dp_0
        d_r2 = dr_1 - dr_0
        d2p_dr2 = d2_p / d_r2
        d_p_d_r = dp_0 / dr_0
        if d_p_d_r > tolerance and d2p_dr2 >= 0:
            radius = radii[counter]
            mass = masses[counter]
            print("Roots Found at R = ", round(radius, 1),
                  " km and M = ", round(mass, 2), " M0")
            return radius, mass

        else:
            counter += 1
    print("No Roots Found")
    return 0, 0


def root_prev(radii, states, tolerance):
    # masses = states[:, 0] # uni rk4
    # pressures = states[:, 1] # uni rk4
    masses, pressures = states  # scipy
    counter = 2
    abs_tol = -1 * pressures[0] * tolerance / radii[-1]
    while counter < len(radii):
        pressure_2 = pressures[counter]
        pressure_1 = pressures[counter - 1]
        pressure_0 = pressures[counter - 2]
        radius_2 = radii[counter]
        radius_1 = radii[counter - 1]
        radius_0 = radii[counter - 2]

        dp_0 = pressure_1 - pressure_0
        dp_1 = pressure_2 - pressure_1
        dr_0 = radius_1 - radius_0
        dr_1 = radius_2 - radius_1
        d2_p = dp_1 - dp_0
        d_r2 = dr_1 - dr_0
        d2p_dr2 = d2_p / d_r2
        d_p_d_r = dp_1 / dr_1
        if d_p_d_r > abs_tol and d2p_dr2 >= 0:
            radius = radii[counter]
            mass = masses[counter]
            print("Roots Found at R = ", round(radius, 1),
                  " km and M = ", round(mass, 1), " M0")
            return radius, mass

        else:
            counter += 1
    print("No Roots Found")
    return 0, 0


def mass_saturation(radii, states, tolerance):

    # code go here
    return None


def iterate(grad, r_0, step, num, min_pressure, max_pressure, pressure_step, tolerance, filename, plot_time, plot_individual):
    whole_start = tm.time()
    pressure = min_pressure
    radii_output = np.zeros((0, 1))
    mass_output = np.zeros((0, 1))
    pressures = np.zeros((0, 1))
    times = np.zeros((0, 1))
    iterations = np.linspace(0, (max_pressure - min_pressure) /
                             pressure_step, int((max_pressure - min_pressure)/pressure_step))
    counter = 0
    while pressure < max_pressure:
        print("Calculating at pressure ", round(pressure, 2), "Pa ...")
        start_time = tm.time()
        # state_0 = np.array([0, pressure]) # uni rk4
        state_0 = [0, pressure]  # scipy
        pressures = np.append(pressures, pressure)
        # radii, states = rk4(grad, r_0, state_0, step, num) # uni rk4
        radii, states = scipy(grad, r_0, state_0, step, num)  # scipy
        # radius, mass = root_next(radii/1000, states, tolerance) # root finding using next points
        # root finding using previous points
        radius, mass = root_prev(radii/1000, states, tolerance)
        radii_output = np.append(radii_output, radius)
        mass_output = np.append(mass_output, mass)
        pressure = pressure + pressure_step
        function_time = tm.time() - start_time
        times = np.append(times, function_time)
        print("Finished calculation ", counter,
              " in ", round(function_time, 1), "s ...")
        if plot_individual:
            plot_root(radii/1000, states, filename + "_Individual", radius)
        counter += 1
    whole_time = tm.time() - whole_start
    print("Computation finished in ", round(whole_time, 1), "s")
    if plot_time:
        plot_times(iterations, times)

    return pressures, radii_output, mass_output


def root(radii, states, tolerance):
    masses, pressures = states
    loc_array = np.where(pressures <= tolerance)
    prime_index = loc_array[0]
    # prime_index = indices[0]
    if len(radii[prime_index]) == 0:
        print("No roots found ...")
        return 0, 0
    else:
        radius, mass, pressure = radii[prime_index][0], masses[prime_index][0], pressures[prime_index]
        print("Root found at", radius /
              1000, "km and", mass, "M0.")
        return radius, mass


def rory(radius, state, tolerance):
    e = np.where(state[1] == min(state[1]))
    i = e[0][0]
    r_val = radius[i]
    m_val = state[0][i]
    return r_val, m_val


if __name__ == "__main__":
    main()
