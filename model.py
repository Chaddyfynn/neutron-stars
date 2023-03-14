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

# RK4 / Calculator Settings
R_0 = 0.0001  # Initial Condition Radius, m
R_F = 20_000_000  # Final Radius, m
NUM = 1_000_000  # Number of Steps
STEP = R_F / NUM  # Step, dx
R_SPAN = [R_0, R_F]  # Radii Span

# System Settings
MIN_PRESSURE = 2.2e21  # Minimum Central Pressure, Pa
MAX_PRESSURE = 4e21  # Maximum Central Pressure, Pa
NUM_STEPS = 10  # Number of Iterations (Plot Points on Graph)
PRESSURE_STEP = (MAX_PRESSURE - MIN_PRESSURE) / NUM_STEPS
TOLERANCE = 0.01  # 0 - 1 factor of max dp/dr root point tolerance (ideal=0)

# Astronomical Constant
M0 = 1.98847e30  # Solar Mass, kg
R0 = (con.G*M0)/(con.c**2)  # Solar Schwarzchild Radius, m

# Save and Graph Settings
FILENAME = "Neutron_Star_Non_Rel_Polytrope"  # Graph and Text File Desired Name
PLOT_TIME = False  # Plot Function Evaluation Times vs Pressure? (Boolean)
# Compute for a range of central pressures (True), or one (False)
FULL_COMPUTATION = False
PLOT_INDIVIDUAL = False  # Create graphs for each central pressure (False)
CROP = 0  # Left Crop for Full computation, 5e23 for rel
METADATA = [R_0, STEP, NUM, MIN_PRESSURE, MAX_PRESSURE, NUM_STEPS]

# Polytropic Constants
K_WD_NON_REL = con.hbar**2/(15*np.pi**2*con.m_e) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(5/3)  # Pressure Constant, No Unis
K_WD_REL = con.hbar * con.c / (12 * np.pi**2) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(4/3)  # Pressure Constant, No Unis
K_N_NON_REL = con.hbar**2/(15*np.pi**2*con.m_n) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(5/3)  # Pressure Constant, No Unis
GAMMA_NON_REL = 5/3  # Polytropic Index, No Units
GAMMA_REL = 4/3


class PolytropeModel:  # Defines the structure of any gas with a Polytropic EoS
    def __init__(self, pressure_constant, polytropic_index):
        self.k = pressure_constant  # Creates k property
        self.gamma = polytropic_index  # Creates gamma property

    def __str__(self):
        # Defines str formatting
        return f"Polytropic model k={self.k} gamma={self.gamma}"

    def energy_density(self, state):  # state = [mass (kg), pressure (Pa)]
        mass, pressure = state
        # Calculates energy density, Pa
        return np.power(pressure / self.k, 1 / self.gamma)

    # radius = [radius (m)], state = [mass (kg), pressure (Pa)]
    def grad(self, radius, state):
        mass, pressure = state
        # Pre-calculates radius^2 constant for efficiency
        sq_radius = np.power(radius, 2)
        # Mass differential Eqn
        dm_dr = ((4 * np.pi * sq_radius) *
                 self.energy_density(state)/(M0*(con.c)**2))
        # Pressure Differential Eqn
        dp_dr = -1 * (R0*mass/sq_radius) * self.energy_density(state)

        # gradient array [mass derivative, pressure derivative]
        return np.array([dm_dr, dp_dr])


class PolyStar(PolytropeModel):  # Defines a star with a polytropic EoS
    def __init__(self, pressure_constant, polytropic_index, central_pressure):
        # Inherit polytropic properties
        super().__init__(pressure_constant, polytropic_index)
        self.p0 = central_pressure  # Create central pressure property

    def __str__(self):
        # String formatting
        return f"Polytropic model k={self.k}, gamma={self.gamma}, p0={self.p0} Pa"

    def increment(self, step):
        self.p0 += step
        return None


def solve_individual(body, r_span):
    state_0 = [0, body.p0]  # Initial state at R_0
    solution = sci_int.solve_ivp(body.grad, r_span, state_0, method='RK45',
                                 t_eval=None, dense_output=False, events=None,
                                 vectorized=True, args=None, max_step=1000)

    radii = solution['t']
    masses = solution['y'][0]
    pressures = solution['y'][1]
    states = masses, pressures
    return radii, states


def root(radii, states, tolerance):
    masses, pressures = states
    loc_array = np.where(pressures <= tolerance)
    prime_index = loc_array[0]
    # prime_index = indices[0]
    radius, mass, pressure = radii[prime_index][0], masses[prime_index], pressures[prime_index]
    print("Root found at", round(radius, 1) /
          1000, "km and", round(mass, 2), "M0.")
    return radius, mass


def solve_range(body, max_pressure, pressure_step, tolerance, r_span, filename):
    radii_1 = np.zeros((0, 1))
    masses = np.zeros((0, 1))
    pressures = np.zeros((0, 1))
    counter = 1
    while body.p0 < max_pressure:
        print("Calculating", counter, " at pressure",
              round(body.p0, 2), "Pa ...")
        start_time = time.time()
        radii_2, states = solve_individual(body, r_span)
        radius, mass = calc.root_prev(radii_2, states, tolerance)
        # radius, mass = root(radii_2, states, tolerance)
        radii_1 = np.append(radii_1, radius)
        masses = np.append(masses, mass)
        pressures = np.append(pressures, body.p0)
        body.increment(pressure_step)
        calc_time = time.time() - start_time
        print("Calculation", counter, "completed in", round(calc_time, 1), "s.")
        counter += 1
        if PLOT_INDIVIDUAL:
            calc.plot_root(radii/1000, states, filename +
                           "_Individual", radius/1000)
    return radii_1, masses, pressures


if __name__ == "__main__":
    non_rel_wd_star = PolyStar(K_WD_NON_REL, GAMMA_NON_REL, MIN_PRESSURE)
    rel_wd_star = PolyStar(K_WD_REL, GAMMA_REL, MIN_PRESSURE)
    non_rel_n_star = PolyStar(K_N_NON_REL, GAMMA_NON_REL, MIN_PRESSURE)
    star = non_rel_wd_star
    if FULL_COMPUTATION:
        radii, masses, pressures = solve_range(
            star, MAX_PRESSURE, PRESSURE_STEP, TOLERANCE, R_SPAN, FILENAME)
        calc.plot_pressure(pressures, radii/1000, masses, FILENAME, CROP)
        states = np.c_[radii/1000, masses]
        calc.save(pressures, states, FILENAME, METADATA)
    else:
        radii, states = solve_individual(star, R_SPAN)
        radius, mass = calc.root_prev(radii, states, TOLERANCE)
        calc.plot_root(radii/1000, states, FILENAME, radius/1000)
        calc.save(radii/1000, states, FILENAME, METADATA)
