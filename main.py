# -*- coding: utf-8 -*-
"""
|/--------------------TITLE--------------------\|
PHYS20161 -- Assignment N -- [ASSIGNMENT NAME]
|/---------------------------------------------\|

[CODE DESCRIPTION/USAGE/OUTLINE]


Created: Fri Mar 10 19:40:31 2023
Last Updated:

@author: Charlie Fynn Perkins, UID: 10839865 0
"""
import numpy as np
import scipy.constants as con
import calculator as calc
import scipy.integrate as sci_int
import time
import plotter as plt
import models
import equation_of_state as eos
import energy_density_multithread as edm
import getopt
import sys

# Numerical Methods / Calculator Settings
R_0 = 0.0001  # Initial Condition Radius, m
R_F = 6_000  # Final Radius, m SUB 10k
R_SPAN = [R_0, R_F]  # Radii Span

# System Settings
MIN_PRESSURE = 1.58e42  # Minimum Central Pressure, Pa 1e28
MAX_PRESSURE = 1e46  # Maximum Central Pressure, Pa 1e46
NUM_STEPS = 20  # Number of Iterations (Plot Points on Graph)
PRESSURE_STEP = (MAX_PRESSURE - MIN_PRESSURE) / NUM_STEPS
LOGARITHMIC = True  # Plot and Produce Points Logarithmically? (Boolean)
if LOGARITHMIC:
    P_EVAL = np.logspace(np.log10(MIN_PRESSURE), np.log10(
        MAX_PRESSURE), NUM_STEPS)  # Logarithmic Points
else:
    P_EVAL = np.linspace(MIN_PRESSURE, MAX_PRESSURE,
                         NUM_STEPS)  # Linear Points

# Radius Root Finding Tolerance (Changes Meaning According to Root Finding Algorithm)
TOLERANCE = 0.0001

# Save and Graph Settings
FILENAME = "REPEAT"  # Graph and Text File Desired Name
PLOT_TIME = False  # Plot Function Evaluation Times vs Pressure? (Boolean)
# Compute for a range of central pressures (True), or one (False)
# Compute over central pressure range? (0 to generate e_dens array)
# True performs max radius & mass over pressure range, 2 generates energy density data, False produces single graph
FULL_COMPUTATION = False
PLOT_INDIVIDUAL = True  # Create graphs for each central pressure (False)
CROP = 0  # Left Crop for Full computation, 5e23 for rel
METADATA = [R_0, MIN_PRESSURE, MAX_PRESSURE,
            NUM_STEPS]  # Desired save metadata

# Polytropic Constants
K_WD_NON_REL = con.hbar**2/(15*np.pi**2*con.m_e) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(5 /
                                          3)  # White Dwarf Non Relativistc Pressure Constant, No Unis
K_WD_REL = con.hbar * con.c / (12 * np.pi**2) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(4 /
                                          3)  # White Dwarf Relativistic Pressure Constant, No Unis
K_N_NON_REL = con.hbar**2/(15*np.pi**2*con.m_n) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(5 /
                                          3)  # Neutron Star Non Relativistic Pressure Constant, No Unis
GAMMA_NON_REL = 5/3  # Non Relativistic Polytropic Index, No Units
GAMMA_REL = 4/3  # Relativistic Polytropic Index, No Units


def solve_individual(body, r_span):
    state_0 = [1e-6, body.p0]  # Initial state at R_0
    start_time = time.time()
    solution = sci_int.solve_ivp(body.grad, r_span, state_0, method='RK45',
                                 t_eval=None, dense_output=False, events=None,
                                 vectorized=True, args=None, max_step=1000)
    print("Calculation finished in", round(time.time() - start_time, 2), "s")
    radii = solution['t']
    #print("Radii", radii)
    masses = solution['y'][0]
    #print("Masses", masses)
    pressures = solution['y'][1]
    #print("Pressures", pressures)
    states = masses, pressures
    return radii, states


def solve_range(body, max_pressure, pressure_step, tolerance, r_span, filename):
    r_0 = r_span[0]
    radii_1 = np.zeros((0, 1))
    masses = np.zeros((0, 1))
    pressures = np.zeros((0, 1))
    counter = 1
    while body.p0 < max_pressure:
        print("Calculating", counter, " at pressure",
              round(body.p0, 2), "Pa ...")
        start_time = time.time()
        radii_2, states = solve_individual(body, r_span)
        radius, mass = calc.root_scale_mass(radii_2, states, tolerance)
        radii_1 = np.append(radii_1, radius)
        masses = np.append(masses, mass)
        pressures = np.append(pressures, body.p0)
        if LOGARITHMIC:
            body.p0 = P_EVAL[counter - 1]
        else:
            body.increment(pressure_step)
        calc_time = time.time() - start_time
        print("Calculation", counter, "completed in", round(calc_time, 1), "s.")
        counter += 1
        if PLOT_INDIVIDUAL:
            calc.plot_root(radii_2/1000, states, filename +
                           "_Individual", radius/1000)
    return radii_1, masses, pressures


if __name__ == "__main__":
    # Remove 1st argument from the
    # list of command line arguments
    argumentList = sys.argv[1:]

    # Options
    options = "mpP:"

    # Long options
    long_options = ["Model", "Min_Pressure", "Max_Pressure"]

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)

        # checking each argument
        for currentArgument, currentValue in arguments:

            if currentArgument in ("-h", "--Help"):
                print("Displaying Help")

            elif currentArgument in ("-m", "--My_file"):
                print("Displaying file_name:", sys.argv[0])

            elif currentArgument in ("-o", "--Output"):
                print(("Enabling special output mode (% s)") % (currentValue))

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))

    star = models.TOVProNeuElec(MIN_PRESSURE)
    if FULL_COMPUTATION is True:
        radii, masses, pressures = solve_range(
            star, MAX_PRESSURE, PRESSURE_STEP, TOLERANCE, R_SPAN, FILENAME)
        calc.plot_pressure(pressures, radii/1000, masses, FILENAME, CROP)
        states = np.c_[radii/1000, masses]
        calc.save(pressures, states, FILENAME, METADATA)
    elif FULL_COMPUTATION == 2:
        start_time = time.time()
        edm.energy_density(MIN_PRESSURE, MAX_PRESSURE, NUM_STEPS)
        print("FINISHED!!! in", round(time.time() - start_time, 0), "s")
    else:
        radii, states = solve_individual(star, R_SPAN)
        radius, mass = calc.root_scale_mass(radii, states, TOLERANCE)
        calc.plot_root(radii/1000, states, FILENAME, radius/1000)
        calc.save(radii/1000, states, FILENAME, METADATA)
