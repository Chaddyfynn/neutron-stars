# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:50:15 2023

@author: Roryt
"""

import RK4_p0_function as solve
import numpy as np
import matplotlib.pyplot as plt


p_max = 1e20
p = 1e25
r_vals = []
m_vals = []

rval, mval = solve.main(p)
print(rval)
print(mval)

# while p < p_max:
#   rval, mval = solve.main(p)
#  r_vals.append(rval)
# m_vals.append(mval)
#p += 2e19
