# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:34:21 2023

@author: Roryt
"""
import numpy as np
import scipy.constants as con
import scipy.integrate as sci_int
import scipy.optimize as sci_opt


r_0 = 0.001
r_max = 12000
t_span = [0.001, 18000000]
state_initial = 2e32/10


M0 = 1.98847e30

R0 = (con.G*M0)/(con.c**2)
K = (con.hbar**2)/(15*np.pi**2*con.m_n) * \
    ((3*np.pi**2)/(con.m_n*con.c**2))**(4/3)

GAMMA = 4/3
eps0 = (con.m_e**4 * con.c**5)/(np.pi**2*con.hbar**3)


def main(state_0):
    solution = sci_int.solve_ivp(grad, t_span, np.array([0, state_0]), method='RK45',
                                 t_eval=None, dense_output=False, events=None,
                                 vectorized=True, args=None, max_step=1000)

    radius = solution['t']
    state = solution['y']

    # if min(state[1]) > 1e-10:
    e = np.where(state[1] == min(state[1]))
    # else:
    #   e = np.where(state[1] < 1e-10)

    i = e[0][0]

    r_val = radius[-1]
    m_val = state[0][-1]

    return r_val, m_val


def p_func(x):
    #x = kf/(con.m_e*con.c)
    # print(x)
    return (eps0/24 * ((2*x**3-3*x)*(1+x**2)**1/2 + 3*np.arcsinh(x)) - state_initial)


def eps_func(x):
    return eps0/8 * ((2*x**3+x)*(1+x**2)**(1/2) - np.arcsinh(x))


def grad(radius, state):
    x0 = 100
    d = sci_opt.newton(p_func, x0, tol=1e-4, maxiter=10000)

    eps = eps_func(d)

    m, p = state

    dm_dr = 4*np.pi*radius**2/con.c**2 * eps
    # dp_dr = -1 * ((con.G*eps*m)/(M0*con.c**2*radius**2))*(1+p/eps)*(1+(4*np.pi *
    #                                                                  radius**3*p*M0)/(m*con.c**2))*(1-(2*con.G*m)/(M0*con.c**2*radius))**-1
    dp_dr = -1 * con.G*m*eps/(con.c**2*radius**2)

    return np.array([dm_dr, dp_dr])


x0 = 100
d = sci_opt.newton(p_func, x0, tol=1e-4, maxiter=10000)

eps = eps_func(d)
print(eps)
print(main(state_initial))
