# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:07:38 2024

@author: snagchowdh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system of differential equations
def system(t, variables, alpha, beta, omega):
    x, y = variables
    dxdt = (alpha - beta * (x**2 + y**2)) * x - omega * y
    dydt = (alpha - beta * (x**2 + y**2)) * y + omega * x
    return [dxdt, dydt]

# Parameters
alpha = 1.0
beta = 1.0
omega = 3.0

# Solve the system using RKF45 method
t_span = (0, 20)
t_eval = np.linspace(*t_span, 2000)
init_conditions = np.random.uniform(-1, 1, size=2)
sol = solve_ivp(system, t_span, init_conditions, args=(alpha, beta, omega), t_eval=t_eval, method='RK45')

# Plot only the last 500 steps of the phase portrait
plt.figure(figsize=(8, 6))
plt.plot(sol.y[0][-500:], sol.y[1][-500:], 'r-', linewidth=0.5)
plt.title('Phase Portrait (Last 500 Steps)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(False)
plt.show()

