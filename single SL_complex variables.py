# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:37:56 2024

@author: snagchowdh
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the function representing the system of ODEs
def z_dot(t, z, alpha, beta, omega):
    return (1j*omega + alpha - beta*(abs(z)**2)) * z

# Define the parameters
alpha = 0.5
beta = 1
omega = 3.0
step_length = 0.01

# Define the time points for integration
t_span = (0, 2000)  # Integration time interval
t_eval = np.arange(t_span[0], t_span[1], step_length)

# Define the function to generate random initial conditions
def random_initial_conditions():
    real_part = np.random.uniform(-1, 1)
    imag_part = np.random.uniform(-1, 1)
    return complex(real_part, imag_part)

# Solve the differential equation for random initial conditions
initial_conditions = random_initial_conditions()
solution = solve_ivp(z_dot, t_span, [initial_conditions], args=(alpha, beta, omega), t_eval=t_eval, method='RK45')

# Extract the solution
z_solution = solution.y[0]
t_solution = solution.t

# Create a subplot with 2 rows and 1 column
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# Plot real part of z for the final iterations
axs[0].plot(t_solution[-1000:], np.real(z_solution[-1000:]), label='Real(z)')
# Plot imaginary part of z for the final iterations
axs[0].plot(t_solution[-1000:], np.imag(z_solution[-1000:]), label='Imaginary(z)')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Value')
axs[0].set_title('Real and Imaginary Parts of z(t) for Final Iterations')
axs[0].legend()
axs[0].grid(True)

# Plot real vs imaginary parts of z
axs[1].plot(np.real(z_solution[-1000:]), np.imag(z_solution[-1000:]), label='Trajectory of z(t)')
axs[1].set_xlabel('Real(z)')
axs[1].set_ylabel('Imaginary(z)')
axs[1].set_title('Trajectory of z(t) in Complex Plane')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

