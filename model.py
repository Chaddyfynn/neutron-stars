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
import scipy.optimize as sci_opt
import time

# RK4 / Calculator Settings
R_0 = 0.0001  # Initial Condition Radius, m
R_F = 5_000  # Final Radius, m
R_SPAN = [R_0, R_F]  # Radii Span

# System Settings
MIN_PRESSURE = 1e29  # Minimum Central Pressure, Pa
MAX_PRESSURE = 1e33  # Maximum Central Pressure, Pa
NUM_STEPS = 100  # Number of Iterations (Plot Points on Graph)
PRESSURE_STEP = (MAX_PRESSURE - MIN_PRESSURE) / NUM_STEPS
TOLERANCE = 1e15  # Radius Root Finding Tolerance (Changes Meaning According to Root Finding Algorithm)
LOGARITHMIC = False # Plot and Produce Points Logarithmically? (Boolean)
if LOGARITHMIC:
    P_EVAL = np.logspace(np.log10(MIN_PRESSURE), np.log10(MAX_PRESSURE), NUM_STEPS) # Logarithmic Points
else:
    P_EVAL = np.linspace(MIN_PRESSURE, MAX_PRESSURE, NUM_STEPS) # Linear Points

# Astronomical Constant
M0 = 1.98847e30  # Solar Mass, kg
R0 = (con.G*M0)/(con.c**2)  # Solar Schwarzchild Radius, m

# Thermodynamic Constants
EPS_N_0 = np.power(con.m_n, 4) * np.power(con.c, 5) / (np.power(np.pi, 2) *
                                                       np.power(con.hbar, 3)) # Neutron Star Energy Density Constant, J m^-3
EPS_E_0 = np.power(con.m_e, 4) * np.power(con.c, 5) / (np.power(np.pi, 2) *
                                                       np.power(con.hbar, 3)) # Neutron Star Energy Density Constant, J m^-3
EPS_P_0 = np.power(con.m_p, 4) * np.power(con.c, 5) / (np.power(np.pi, 2) *
                                                       np.power(con.hbar, 3)) # Neutron Star Energy Density Constant, J m^-3

# Save and Graph Settings
FILENAME = "TOV_Neutron_Star_Range"  # Graph and Text File Desired Name
PLOT_TIME = False  # Plot Function Evaluation Times vs Pressure? (Boolean)
# Compute for a range of central pressures (True), or one (False)
FULL_COMPUTATION = False # Compute over central pressure range?
PLOT_INDIVIDUAL = True  # Create graphs for each central pressure (False)
CROP = 0  # Left Crop for Full computation, 5e23 for rel
METADATA = [R_0, MIN_PRESSURE, MAX_PRESSURE, NUM_STEPS] # Desired save metadata

# Polytropic Constants
K_WD_NON_REL = con.hbar**2/(15*np.pi**2*con.m_e) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(5/3)  # White Dwarf Non Relativistc Pressure Constant, No Unis
K_WD_REL = con.hbar * con.c / (12 * np.pi**2) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(4/3)  # White Dwarf Relativistic Pressure Constant, No Unis
K_N_NON_REL = con.hbar**2/(15*np.pi**2*con.m_n) * \
    ((3*np.pi**2)/(2*con.m_n*con.c**2))**(5/3)  # Neutron Star Non Relativistic Pressure Constant, No Unis
GAMMA_NON_REL = 5/3  # Non Relativistic Polytropic Index, No Units
GAMMA_REL = 4/3 # Relativistic Polytropic Index, No Units


class Star(): # Star Superclass
    def __init__(self, central_pressure):
        self.p0 = central_pressure # Create p0 attribute

    def __str__(self): # String formatting
        return f"Generic Star: p0={self.p0} Pa"

    def increment(self, step): # Central Pressure Increment Function
        self.p0 += step # Increase p0 by step
        return None


class Newtonian(Star): # Newtonian Sub and Superclass
    def __init__(self, central_pressure): 
        super().__init__(central_pressure) # Inherit Star

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


class PureNeutronFermiModel(Newtonian):
    def __init__(self, central_pressure):
        super().__init__(central_pressure)
    
    def fermi_pressure(self, x, pressure):
        # print("Calculating at", pressure)
        return EPS_N_0/24 * ((2 * np.power(x, 3) - 3 * x) *
                             np.power(1 + np.power(x, 2), 1/2) + 3 * np.arcsinh(x)) - \
            pressure


    def fermi_energy_density(self, x):
        return EPS_N_0/8 * ((2 * np.power(x, 3) + x) * np.power(1 +
                                                                np.power(x, 2), 1/2) - np.arcsinh(x))

    def energy_density(self, state):
        mass, pressure = state
        x = sci_opt.bisect(self.fermi_pressure, 150, -150,
                           args=(pressure), maxiter=10000)
        return self.fermi_energy_density(x)
    
class ProtonElectronNeutronFermiModel(Newtonian):
    def __init__(self, central_pressure):
        super().__init__(central_pressure)
        self.last_solution = [0, 0, 0]
        
    def proton_pressure(self, x):
        return EPS_P_0/24 * ((2 * np.power(x, 3) - 3 * x) *
                             np.power(1 + np.power(x, 2), 1/2) + 3 * np.arcsinh(x))
        
    def neutron_pressure(self, x):
        return EPS_N_0/24 * ((2 * np.power(x, 3) - 3 * x) *
                             np.power(1 + np.power(x, 2), 1/2) + 3 * np.arcsinh(x))
        
    def electron_pressure(self, x):
        return EPS_E_0/24 * ((2 * np.power(x, 3) - 3 * x) *
                             np.power(1 + np.power(x, 2), 1/2) + 3 * np.arcsinh(x))
    
    def fermi_pressure(self, x):
        return self.proton_pressure(x[0]) + self.neutron_pressure(x[1]) + self.electron_pressure(x[2])
    
    def fermi_pressure_calc(self, x, pressure):
        scalar = self.fermi_pressure(x) - pressure
        scalar = scalar[0]
        return np.array([scalar, scalar, scalar])
    
    def proton_energy_density(self, x):
        return EPS_P_0/8 * ((2 * np.power(x, 3) + x) * np.power(1 +
                                                                np.power(x, 2), 1/2) - np.arcsinh(x))
        
    def neutron_energy_density(self, x):
        return EPS_N_0/8 * ((2 * np.power(x, 3) + x) * np.power(1 +
                                                                np.power(x, 2), 1/2) - np.arcsinh(x))
        
    def electron_energy_density(self, x):
        return EPS_E_0/8 * ((2 * np.power(x, 3) + x) * np.power(1 +
                                                                np.power(x, 2), 1/2) - np.arcsinh(x))
    
    def energy_density(self, x):
        return self.proton_energy_density(x[0]) + self.neutron_energy_density(x[1]) + self.electron_energy_density(x[2])

    def energy_density_calc(self, state):
        mass, pressure = state
        solution = sci_opt.root(self.fermi_pressure_calc, self.last_solution,
                           args=(pressure), method='broyden1', jac=False) # broyden1,2, anderson okay but slow, df-sane okay but different to broyden
        x = solution.x
        print("Solution Found", x)
        self.last_solution = x
        return self.energy_density(x)


# Defines the structure of any gas with a Polytropic EoS
class PolytropeModel(Newtonian):
    def __init__(self, pressure_constant, polytropic_index, central_pressure):
        super().__init__(central_pressure)
        self.k = pressure_constant  # Creates k property
        self.gamma = polytropic_index  # Creates gamma property

    def __str__(self):
        # Defines str formatting
        return f"Polytropic Star: k={self.k} gamma={self.gamma}, p0={self.p0}"

    def energy_density(self, state):  # state = [mass (kg), pressure (Pa)]
        mass, pressure = state
        # Calculates energy density, Pa
        return np.power(pressure / self.k, 1 / self.gamma)

    # radius = [radius (m)], state = [mass (kg), pressure (Pa)]


class TOVModel(PureNeutronFermiModel):
    def __init__(self, central_pressure):
        super().__init__(central_pressure)
        # Code go here

    def grad(self, radius, state):
        # print("Calculating Grad at", round(radius, 0), "m")
        mass, pressure = state
        sq_radius = np.power(radius, 2)
        cb_radius = np.power(radius, 3)
        e_dens = self.full_energy_density(state)
        c_2 = np.power(con.c, 2)
        term_3 = np.power(1 - 2 * con.G * (mass) / (c_2 * radius), -1)
        if term_3 < 0:
            print("Imaginary Solution")
            return np.array([0, 0])
        dp_dr = -1 * con.G * e_dens * (mass) / (c_2 * sq_radius) * \
            (1 + pressure/e_dens) * (1 + (4 * np.pi * cb_radius * pressure)/((mass) * c_2)) * \
            term_3
        dm_dr = ((4 * np.pi * sq_radius) * e_dens/((con.c)**2))
        return np.array([dm_dr, dp_dr])
    
class TOVModel2(ProtonElectronNeutronFermiModel):
    def __init__(self, central_pressure):
        super().__init__(central_pressure)
        # Code go here

    def grad(self, radius, state):
        # print("Calculating Grad at", round(radius, 0), "m")
        mass, pressure = state
        sq_radius = np.power(radius, 2)
        cb_radius = np.power(radius, 3)
        e_dens = self.energy_density_calc(state)
        c_2 = np.power(con.c, 2)
        term_3 = np.power(1 - 2 * con.G * (mass) / (c_2 * radius), -1)
        if term_3 < 0:
            print("Imaginary Solution")
            return np.array([0, 0])
        dp_dr = -1 * con.G * e_dens * (mass) / (c_2 * sq_radius) * \
            (1 + pressure/e_dens) * (1 + (4 * np.pi * cb_radius * pressure)/((mass) * c_2)) * \
            term_3
        dm_dr = ((4 * np.pi * sq_radius) * e_dens/((con.c)**2))
        return np.array([dm_dr, dp_dr])


def solve_individual(body, r_span):
    state_0 = [1e-6, body.p0]  # Initial state at R_0
    start_time = time.time()
    solution = sci_int.solve_ivp(body.grad, r_span, state_0, method='RK45',
                                 t_eval=None, dense_output=False, events=None,
                                 vectorized=True, args=None, max_step=1000)
    print("Calculation finished in", round(time.time() - start_time, 2), "s")
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
        # radius, mass = calc.root_prev(radii_2, states, TOLERANCE)
        radius, mass = root(radii_2, states, tolerance)
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
            calc.plot_root(radii/1000, states, filename +
                           "_Individual", radius/1000)
    return radii_1, masses, pressures


if __name__ == "__main__":
    non_rel_wd_star = PolytropeModel(
        K_WD_NON_REL, GAMMA_NON_REL, MIN_PRESSURE)
    rel_wd_star = PolytropeModel(K_WD_REL, GAMMA_REL, MIN_PRESSURE)
    non_rel_n_star = PolytropeModel(K_N_NON_REL, GAMMA_NON_REL, MIN_PRESSURE)
    full_tov_n_star = TOVModel(MIN_PRESSURE)
    tov_n_star = TOVModel2(MIN_PRESSURE)
    full_newton_n_star = PureNeutronFermiModel(MIN_PRESSURE)
    star = tov_n_star
    if FULL_COMPUTATION:
        radii, masses, pressures = solve_range(
            star, MAX_PRESSURE, PRESSURE_STEP, TOLERANCE, R_SPAN, FILENAME)
        calc.plot_pressure(pressures, radii/1000, masses, FILENAME, CROP)
        states = np.c_[radii/1000, masses]
        calc.save(pressures, states, FILENAME, METADATA)
    else:
        radii, states = solve_individual(star, R_SPAN)
        # radius, mass = rory(radii, states, TOLERANCE)
        radius, mass = root(radii, states, TOLERANCE)
        # radius, mass = calc.root_prev(radii, states, TOLERANCE)
        calc.plot_root(radii/1000, states, FILENAME, radius/1000)
        calc.save(radii/1000, states, FILENAME, METADATA)
