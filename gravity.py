# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:05:33 2023

@author: chadd
"""
from star import Star
from equation_of_state import PolytropeModel, PureNeutronFermiModel, ProtonElectronNeutronFermiModel
import numpy as np
import scipy.constants as con

# Astronomical Constant
M0 = 1.98847e30  # Solar Mass, kg
R0 = (con.G*M0)/(con.c**2)  # Solar Schwarzchild Radius, m

class NewtonianPolytrope(PolytropeModel):  # Newtonian Sub and Superclass
    def __init__(self, central_pressure, polytropic_constant, polytropic_index):
        super().__init__()  # Inherit Star

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
    
class Newtonian(Star):  # Newtonian Sub and Superclass
    def __init__(self, central_pressure):
        super(Newtonian, self).__init__(central_pressure)  # Inherit Star

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

class Newtonian(Star):  # Newtonian Sub and Superclass
    def __init__(self, central_pressure):
        super(Newtonian, self).__init__(central_pressure)  # Inherit Star

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


class TOVModel(Star):
    def __init__(self, central_pressure):
        super(TOVModel, self).__init__(central_pressure)
        # Code go here

    def grad(self, radius, state):
        # print("Calculating Grad at", round(radius, 0), "m")
        mass, pressure = state
        sq_radius = np.power(radius, 2)
        cb_radius = np.power(radius, 3)
        e_dens = self.energy_density_calc(state)
        if e_dens == 0:
            e_dens = 1e-11
        if mass == 0:
            mass = 1e-11
        c_2 = np.power(con.c, 2)
        term_3 = np.power(1 - 2 * con.G * (mass) / (c_2 * radius), -1)
        if term_3 < 0:
            print("Imaginary Solution")
            # return np.array([0, 0])
        dp_dr = -1 * con.G * e_dens * (mass) / (c_2 * sq_radius) * \
            (1 + pressure/e_dens) * (1 + (4 * np.pi * cb_radius * pressure)/((mass) * c_2)) * \
            term_3
        dm_dr = ((4 * np.pi * sq_radius) * e_dens/((con.c)**2))
        return np.array([dm_dr, dp_dr])
    
class TOVModel(Star):
    def __init__(self, central_pressure):
        super(TOVModel, self).__init__(central_pressure)
        # Code go here

    def grad(self, radius, state):
        # print("Calculating Grad at", round(radius, 0), "m")
        mass, pressure = state
        sq_radius = np.power(radius, 2)
        cb_radius = np.power(radius, 3)
        e_dens = self.energy_density_calc(state)
        if e_dens == 0:
            e_dens = 1e-11
        if mass == 0:
            mass = 1e-11
        c_2 = np.power(con.c, 2)
        term_3 = np.power(1 - 2 * con.G * (mass) / (c_2 * radius), -1)
        if term_3 < 0:
            print("Imaginary Solution")
            # return np.array([0, 0])
        dp_dr = -1 * con.G * e_dens * (mass) / (c_2 * sq_radius) * \
            (1 + pressure/e_dens) * (1 + (4 * np.pi * cb_radius * pressure)/((mass) * c_2)) * \
            term_3
        dm_dr = ((4 * np.pi * sq_radius) * e_dens/((con.c)**2))
        return np.array([dm_dr, dp_dr])

class TOVModel(Star):
    def __init__(self, central_pressure):
        super(TOVModel, self).__init__(central_pressure)
        # Code go here

    def grad(self, radius, state):
        # print("Calculating Grad at", round(radius, 0), "m")
        mass, pressure = state
        sq_radius = np.power(radius, 2)
        cb_radius = np.power(radius, 3)
        e_dens = self.energy_density_calc(state)
        if e_dens == 0:
            e_dens = 1e-11
        if mass == 0:
            mass = 1e-11
        c_2 = np.power(con.c, 2)
        term_3 = np.power(1 - 2 * con.G * (mass) / (c_2 * radius), -1)
        if term_3 < 0:
            print("Imaginary Solution")
            # return np.array([0, 0])
        dp_dr = -1 * con.G * e_dens * (mass) / (c_2 * sq_radius) * \
            (1 + pressure/e_dens) * (1 + (4 * np.pi * cb_radius * pressure)/((mass) * c_2)) * \
            term_3
        dm_dr = ((4 * np.pi * sq_radius) * e_dens/((con.c)**2))
        return np.array([dm_dr, dp_dr])
    
    