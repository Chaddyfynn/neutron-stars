# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:09:35 2023

@author: chadd
"""

import numpy as np
import scipy.constants as con
import scipy.optimize as sci_opt
import time
import calculator as calc

# Thermodynamic Constants
EPS_N_0 = np.power(con.m_n, 4) * np.power(con.c, 5) / (np.power(np.pi, 2) *
                                                       np.power(con.hbar, 3))  # Neutron Star Energy Density Constant, J m^-3
EPS_E_0 = np.power(con.m_e, 4) * np.power(con.c, 5) / (np.power(np.pi, 2) *
                                                       np.power(con.hbar, 3))  # Neutron Star Energy Density Constant, J m^-3
EPS_P_0 = np.power(con.m_p, 4) * np.power(con.c, 5) / (np.power(np.pi, 2) *
                                                       np.power(con.hbar, 3))  # Neutron Star Energy Density Constant, J m^-3

class PureNeutronFermiModel():
    def __init__(self):
        super(PureNeutronFermiModel, self).__init__()
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


class ProtonElectronNeutronFermiModel():
    def __init__(self):
        super(ProtonElectronNeutronFermiModel, self).__init__()
        self.last_solution = [0, 0, 0]
        self.e_dens_array = np.genfromtxt(".\\important_saves\\energy_function_combined_append.txt", dtype=float,
                                          comments='#', delimiter=',', skip_header=1)

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
        scalar = scalar
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
        potential_solution = self.efficient_energy_density_calc(state)
        if potential_solution == "No Solution":
            print("No solution in self.e_dens_array")
            solution = sci_opt.root(self.fermi_pressure_calc, self.last_solution,
                                    args=(pressure), method='broyden1', jac=False)  # broyden1,2, anderson okay but slow, df-sane okay but different to broyden
            x = solution.x
            e_dens = self.energy_density(x)
            self.last_solution = x
            file = open(
                ".\\important_saves\\energy_function_combined_append.txt", 'a')
            output_array = np.c_[state[1], e_dens]
            print(output_array)
            np.savetxt(file, output_array, delimiter=",")
            file.close()
        else:
            print("Table solution exists")
            e_dens = potential_solution
        return e_dens

    def efficient_energy_density_calc(self, state):
        truths = np.logical_and(self.e_dens_array[:, 0] <= (
            state[1] + 1e27), self.e_dens_array[:, 0] >= (state[1] - 1e27))
        if np.any(truths):
            indices = np.where(truths)
            index = indices[int(round(len(indices)/2, 0))]
            e_dens = self.e_dens_array[index, 1]
            if len(e_dens) > 1:
                e_dens = e_dens[0]
            return e_dens
        else:
            return "No Solution"

# Defines the structure of any gas with a Polytropic EoS (White Dwarf)


class PolytropeModel():
    def __init__(self, pressure_constant, polytropic_index):
        super(PolytropeModel, self).__init__()
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
    

def energy_function():
    model = ProtonElectronNeutronFermiModel(0)
    init_pressure = float(0)
    final_pressure = float(1e29)
    step = float(1e25)
    num = (final_pressure - init_pressure) / step
    state = np.array([0, init_pressure])
    pressures = np.zeros((0, 1))
    energy_densities = np.zeros((0, 1))
    momenta = np.zeros((0, 3))
    counter = 1
    init_time = time.time()
    while state[1] <= final_pressure:
        start_time = time.time()
        pressures = np.append(pressures, state[1])
        energy_density = model.energy_density_calc(state)
        energy_densities = np.append(
            energy_densities, energy_density)
        state[1] += step
        print("Calculation", counter, "completed in",
              round(time.time() - start_time, 2), "s")
        print("Time elapsed:", round(time.time() - init_time, 2), "s")
        # print(round(counter*100/num, 2), "% complete")
        counter += 1
    calc.save(pressures, energy_densities, "energy_function", [])