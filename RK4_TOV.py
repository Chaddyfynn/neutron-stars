# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:34:21 2023

@author: Roryt
"""
import numpy as np
import scipy.constants as con
import scipy.integrate as sci_int


r_0 = 0.001
r_max = 12000
t_span = [0.001, 18000]
state_initial = 2e32/10


M0 = 1.98847e30

R0 = (con.G*M0)/(con.c**2)
K = (con.hbar**2)/(15*np.pi**2*con.m_n) * \
    ((3*np.pi**2)/(con.m_n*con.c**2))**(5/3)
GAMMA = 5/3


def main(state_0):
    solution = sci_int.solve_ivp(grad, t_span, np.array([0, state_0]), method='RK45',
                                 t_eval=None, dense_output=False, events=None,
                                 vectorized=True, args=None, max_step=1000)

    radius = solution['t']
    state = solution['y']
    print(radius)

    # if min(state[1]) > 1e-10:
    e = np.where(state[1] == min(state[1]))
    # else:
    #   e = np.where(state[1] < 1e-10)

    i = e[0][0]

    r_val = radius[i]
    m_val = state[0][i]

    return r_val, m_val


def grad(radius, state):
    m, p = state
    eps = np.power((p/K), (1/GAMMA))
    dm_dr = ((4 * np.pi * np.power((radius), 2) *
             eps)/(M0*(con.c)**2))
    dp_dr = -1 * (R0*m*eps/(np.power((radius), 2)))*(1+p/eps)*(1+(4*np.pi *
                                                                  radius**3*p*M0)/(m*con.c**2))*(1-(2*con.G*m)/(M0*con.c**2*radius))**-1

    return np.array([dm_dr, dp_dr])


print(main(state_initial))
