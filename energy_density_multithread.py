# -*- coding: utf-8 -*-
"""
|/--------------------TITLE--------------------\|
PHYS20161 -- Assignment N -- [ASSIGNMENT NAME]
|/---------------------------------------------\|

[CODE DESCRIPTION/USAGE/OUTLINE]


Created: Sun Mar 26 22:27:31 2023
Last Updated:

@author: Charlie Fynn Perkins, UID: 10839865 0
"""
import equation_of_state as eos
from multiprocessing import Pool
import calculator as calc
import numpy as np


def energy_density(init=1e24, fin=1e34, num=100):
    init = float(init)
    fin = float(fin)
    p_evals = np.logspace(np.log10(init), np.log10(fin), num)
    with Pool() as pool:
        energy_densities = pool.map(eos.loop, p_evals)
    energy_densities = np.array(energy_densities)
    calc.save(p_evals, energy_densities, "energy_density_mk2", [])
