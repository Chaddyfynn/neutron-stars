# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:01:37 2023

@author: Roryt
"""

import numpy as np
import scipy.constants as con
import scipy.integrate as sci_int


r_0 = 0.001
r_max = 12000
t_span = [0.001, 18000000]
state_initial = 2.2e22/10


M0 = 1.98847e30

R0 = (con.G*M0)/(con.c**2)

# K = con.hbar**2/(15*np.pi**2*con.m_e) * \
#   ((3*np.pi**2)/(2*con.m_n*con.c**2))**(5/3)

K = (con.hbar*con.c)/(12*np.pi**2) * ((3*np.pi**2)/(2*con.m_n*con.c**2))**(4/3)

GAMMA = 4/3
eps0 = (con.m_e**4 * con.c**5)/(np.pi**2*con.hbar**3)


def main(state_0):
    solution = sci_int.solve_ivp(grad, t_span, np.array([0, state_0]), method='RK45',
                                 t_eval=None, dense_output=False, events=None,
                                 vectorized=True, args=None, max_step=1000)
    # print(solution)
    radius = solution['t']
    state = solution['y']

    #rho = state[0] / (4/3 * np.pi * radius**3)
    #x0 = 1
    #d = sci_opt.newton(func, x0, tol=1e-4, maxiter=10000)

    if min(state[1]) > 1e-10:
        e = np.where(state[1] == min(state[1]))
    else:
        e = np.where(state[1] < 1e-10)

    i = e[0][0]

    r_val = radius[i]
    m_val = state[0][i]
    #p_val = state[1][i]

    return r_val, m_val


def grad(radius, state):

    m, p = state
    dm_dr = ((4 * np.pi * np.power((radius), 2) *
             np.power((p/K), 1/GAMMA))/(M0*(con.c)**2))
    dp_dr = -1 * (R0*m/(np.power((radius), 2))) * \
        np.power((p/K), (1/GAMMA))

    return np.array([dm_dr, dp_dr])
