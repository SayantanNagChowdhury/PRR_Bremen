# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:28:56 2024

@author: snagchowdh
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import solve_ivp

# Define kronecker delta function
def kronecker_delta(i, j):
    return 1 if i == j else 0

# Define Pauli matrices
sigma_x = np.array([[0, 1],
                    [1, 0]])

sigma_y = np.array([[0, -1j],
                    [1j, 0]])

sigma_z = np.array([[1, 0],
                    [0, -1]])

# Define identity matrix
I = np.eye(2)

# Define lattice site shifts as lists of tuples
ex = [(1, 0)]
ey = [(0, 1)]

# Define the dynamics function
def dynamics(t, z_flat, omega, alpha, beta, u, lattice_size):
    z = z_flat.reshape((lattice_size[0], lattice_size[1], 4))
    result = np.zeros_like(z, dtype=np.complex128)
    for x in range(lattice_size[0]):
        for y in range(lattice_size[1]):
            for j in range(4):
                # Compute the first term
                term1 = (1j * omega[x, y, j] + alpha - beta * abs(z[x, y, j])**2) * z[x, y, j]

                # Compute the second term
                term2 = 0
                for k in range(4):
                    valid_indices = (0 <= x + ex[0][0] < lattice_size[0]) & \
                                    (0 <= y + ex[0][1] < lattice_size[1]) & \
                                    (0 <= x + ey[0][0] < lattice_size[0]) & \
                                    (0 <= y + ey[0][1] < lattice_size[1]) & \
                                    (0 <= x - ex[0][0] < lattice_size[0]) & \
                                    (0 <= y - ex[0][1] < lattice_size[1]) & \
                                    (0 <= x - ey[0][0] < lattice_size[0]) & \
                                    (0 <= y - ey[0][1] < lattice_size[1])
                    sum_term = 0
                    for idx, (x_prime, y_prime) in enumerate([(x + ex[0][0], y + ex[0][1]), (x - ex[0][0], y - ex[0][1]), 
                                                               (x + ey[0][0], y + ey[0][1]), (x - ey[0][0], y - ey[0][1])]):
                        if valid_indices:
                            sum_term += h(j, k, [x, y], [x_prime, y_prime]) * z[x_prime, y_prime, k]
                    term2 -= 1j * sum_term

                result[x, y, j] = term1 + term2

    return result.flatten()

def h(j, k, xy, xy_prime):
    x, y = xy
    x_prime, y_prime = xy_prime
    ex = (1, 0)
    ey = (0, 1)
    delta_x = kronecker_delta(x, x_prime)
    delta_y = kronecker_delta(y, y_prime)
    delta_x_plus_ex = kronecker_delta((x + ex[0]) % lattice_size[0], x_prime)
    delta_x_minus_ex = kronecker_delta((x - ex[0]) % lattice_size[0], x_prime)
    delta_y_plus_ey = kronecker_delta((y + ey[1]) % lattice_size[1], y_prime)
    delta_y_minus_ey = kronecker_delta((y - ey[1]) % lattice_size[1], y_prime)

    # Additional terms
    term1 = (u * delta_x * delta_y + 0.5 * (delta_x_plus_ex + delta_x_minus_ex + delta_y_plus_ey + delta_y_minus_ey)) * np.kron(I, sigma_z)[j, k]
    term2 = (1 / (2j)) * (delta_y_plus_ey - delta_y_minus_ey) * np.kron(I, sigma_y)[j, k]
    term3 = (1 / (2j)) * (delta_x_plus_ex - delta_x_minus_ex) * np.kron(sigma_z, sigma_x)[j, k]
    term4 = (1j)* b * delta_x * delta_y * np.kron(sigma_x, sigma_x)[j, k]
    
    # term1 = (-1.0) * (sigma_y) * (u * delta_x * delta_y + 0.5 * (delta_x_plus_ex + delta_x_minus_ex + delta_y_plus_ey + delta_y_minus_ey)) * np.kron(I, sigma_z)[j, k]
    # term2 = (1j) * sigma_y * (0.5) * (delta_y_plus_ey - delta_y_minus_ey) * np.kron(I, sigma_y)[j, k]
    # term3 = (1j) * sigma_y * (0.5) * (delta_x_plus_ex - delta_x_minus_ex) * np.kron(sigma_z, sigma_x)[j, k]
    # term4 = (-1j) * sigma_y * b * delta_x * delta_y * np.kron(sigma_x, sigma_x)[j, k]

    return term1 + term2 + term3 + term4


# Initialize simulation parameters
lattice_size = (3, 3)  # Lattice size
omega_0 = 1.0
delta_omega = 0.2
omega = omega_0 + delta_omega * np.random.randn(lattice_size[0], lattice_size[1], 4)
u = -1
b = 0.5
alpha = 0.5
beta = 1

# Initialize z with real and imaginary parts randomly between -0.1 and 0.1
z_real = np.random.uniform(-0.1, 0.1, size=(lattice_size[0], lattice_size[1], 4))
z_imag = np.random.uniform(-0.1, 0.1, size=(lattice_size[0], lattice_size[1], 4))
z_init = z_real + 1j * z_imag

# Integrate the differential equations using RK45 method with duration=200 and dt=0.005
duration = 100 #200
dt = 0.005
num_steps = int(duration / dt)
t_eval = np.linspace(0, duration, num_steps)
sol = solve_ivp(dynamics, [0, duration], z_init.flatten(), method='RK45', t_eval=t_eval, args=(omega, alpha, beta, u, lattice_size))

# Extract the amplitudes of the first oscillator at the final time
amplitudes = np.abs(sol.y[:, -1].reshape((lattice_size[0], lattice_size[1], 4))[:, :, 0])

# Visualize the amplitudes over the lattice
plt.figure(figsize=(6, 6))
plt.imshow(amplitudes, cmap='viridis', origin='upper', extent=(0, lattice_size[1], lattice_size[0], 0))
plt.colorbar(label='Amplitude')
plt.title('Amplitude of the First Oscillator at Final Time')
plt.xlabel('y')
plt.ylabel('x')
plt.grid(False)
plt.show()

# Visualize lattice graph
G = nx.grid_2d_graph(lattice_size[0], lattice_size[1])
pos = {(x, y): (y, -x) for x, y in G.nodes()}
plt.figure(figsize=(6, 6))
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
plt.title('Lattice Graph')
plt.show()
