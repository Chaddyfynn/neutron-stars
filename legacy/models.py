# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:12:59 2023

@author: chadd
"""

import gravity as gr
import equation_of_state as eos

class NewtonianPolytrope(gr.Newtonian, eos.PolytropeModel):
    def __init__(self, central_pressure, pressure_constant, polytropic_index):
        super(NewtonianPolytrope, self).__init__(central_pressure)

class NewtonianPureNeutron(gr.Newtonian, eos.PureNeutronFermiModel):
    def __init__(self, central_pressure):
        super(NewtonianPureNeutron, self).__init__(central_pressure)

class TOVPolytrope(gr.TOVModel, eos.PolytropeModel):
    def __init__(self, central_pressure, pressure_constant, polytropic_index):
        super(TOVPolytrope, self).__init__(central_pressure)

class TOVPureNeutron(gr.TOVModel, eos.PureNeutronFermiModel):
    def __init__(self, central_pressure):
        super(TOVPureNeutron, self).__init__(central_pressure)

class TOVProNeuElec(gr.TOVModel, eos.ProtonElectronNeutronFermiModel):
    def __init__(self, central_pressure):
        super(TOVProNeuElec, self).__init__(central_pressure)
        
