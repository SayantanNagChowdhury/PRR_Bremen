# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:16:22 2024

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

# Number of systems
N = 10

# Solve the systems using RKF45 method
t_span = (0, 2000)
t_eval = np.linspace(*t_span, 200000)

plt.figure(figsize=(12, 6))

for i in range(N):
    # Random initial conditions for each system
    init_conditions = np.random.uniform(-1, 1, size=2)
    # Random omega for each system from normal distribution (0, 1)
    omega = 3.0 #np.random.normal(0, 1)
    
    sol = solve_ivp(system, t_span, init_conditions, args=(alpha, beta, omega), t_eval=t_eval, method='RK45')
    
    # Plot phase portrait
    plt.subplot(1, 2, 1)
    plt.plot(sol.y[0][-1000:], sol.y[1][-1000:], linewidth=0.5)

    plt.title('Phase Portrait (Last 1000 Steps) - {} Systems'.format(N))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(False)

    # Plot t vs x
    plt.subplot(1, 2, 2)
    plt.plot(sol.t[-1000:], sol.y[0][-1000:], linewidth=0.5)
    plt.title('$t$ vs $x$ - {} Systems'.format(N))
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.grid(True)

plt.tight_layout()
plt.show()

