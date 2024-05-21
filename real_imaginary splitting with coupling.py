# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:30:52 2024

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
                    [0, 1]])

# Define identity matrix
I = np.eye(2)

# Define lattice site shifts as lists of tuples
ex = [(1, 0)]
ey = [(0, 1)]

# Define the dynamics function
def dynamics(t, z_flat, omega, alpha, beta, u, b, lattice_size):
    z = z_flat.reshape((lattice_size[0], lattice_size[1], 4))
    result = np.zeros_like(z, dtype=np.complex128)
    for x in range(lattice_size[0]):
        for y in range(lattice_size[1]):
            for j in range(4):
                z_r = np.real(z[x, y, j])
                z_i = np.imag(z[x, y, j])
                
                term1_r = alpha * z_r - beta * (z_r**2 + z_i**2) * z_r - omega[x, y, j] * z_i
                term1_i = alpha * z_i - beta * (z_r**2 + z_i**2) * z_i + omega[x, y, j] * z_r

                term1 = term1_r + 1j * term1_i

                # Compute the second term
                term2_r = 0
                term2_i = 0
                for k in range(4):
                    valid_indices = (0 <= x + ex[0][0] < lattice_size[0]) & \
                                    (0 <= y + ex[0][1] < lattice_size[1]) & \
                                    (0 <= x + ey[0][0] < lattice_size[0]) & \
                                    (0 <= y + ey[0][1] < lattice_size[1]) & \
                                    (0 <= x - ex[0][0] < lattice_size[0]) & \
                                    (0 <= y - ex[0][1] < lattice_size[1]) & \
                                    (0 <= x - ey[0][0] < lattice_size[0]) & \
                                    (0 <= y - ey[0][1] < lattice_size[1])
                    sum_term_r = 0
                    sum_term_i = 0
                    for idx, (x_prime, y_prime) in enumerate([(x + ex[0][0], y + ex[0][1]), (x - ex[0][0], y - ex[0][1]), 
                                                               (x + ey[0][0], y + ey[0][1]), (x - ey[0][0], y - ey[0][1])]):
                        if valid_indices:
                            h_r, h_i = h(j, k, [x, y], [x_prime, y_prime], u, b)
                            z_prime_r = np.real(z[x_prime, y_prime, k])
                            z_prime_i = np.imag(z[x_prime, y_prime, k])
                            sum_term_r += h_r * z_prime_r - h_i * z_prime_i
                            sum_term_i += h_r * z_prime_i + h_i * z_prime_r
                    term2_r += sum_term_i
                    term2_i += -sum_term_r


                term2 = term2_r + 1j * term2_i

                result[x, y, j] = term1 + term2

    return result.flatten()

def h(j, k, xy, xy_prime, u, b):
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
    term1_r = (u * delta_x * delta_y + 0.5 * (delta_x_plus_ex + delta_x_minus_ex + delta_y_plus_ey + delta_y_minus_ey)) * np.kron(I, sigma_z)[j, k]
    term1_i = 0
    term2_r = 0
    term2_i = (1 / 2) * (delta_y_plus_ey - delta_y_minus_ey) * np.kron(I, sigma_y)[j, k]
    term3_r = 0
    term3_i = (1 / 2) * (delta_x_plus_ex - delta_x_minus_ex) * np.kron(sigma_z, sigma_x)[j, k]
    term4_r = 0
    term4_i = b * delta_x * delta_y * np.kron(sigma_x, sigma_x)[j, k]
    
    h_r = term1_r + term2_r + term3_r + term4_r
    h_i = term1_i + term2_i + term3_i + term4_i
    
    return h_r, h_i

# Initialize simulation parameters
lattice_size = (4, 4)  # Lattice size
omega_0 = 1.0
delta_omega = 0.0 #0.2
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
duration = 100  # 200
dt = 0.005
num_steps = int(duration / dt)
t_eval = np.linspace(0, duration, num_steps)
sol = solve_ivp(dynamics, [0, duration], z_init.flatten(), method='RK45', t_eval=t_eval, args=(omega, alpha, beta, u, b, lattice_size))

# Extract the amplitudes of the first oscillator at the final time
amplitudes = np.abs(sol.y[:, -1].reshape((lattice_size[0], lattice_size[1], 4))[:, :, 0])

# Extract the phases of the first oscillator
phases_final = np.angle(sol.y[:, -1].reshape((lattice_size[0], lattice_size[1], 4))[:, :, 0])
phases_final = np.mod(phases_final, 2 * np.pi)  # Ensure phases are within [0, 2π]

# Calculate the instantaneous frequencies
phases = np.angle(sol.y.reshape((lattice_size[0], lattice_size[1], 4, num_steps))[:, :, 0, :])
instantaneous_frequencies = np.diff(phases, axis=-1) / dt
instantaneous_frequencies = np.mod(instantaneous_frequencies + np.pi, 2 * np.pi) - np.pi  # Wrap around [-π, π]

# Time-average the frequencies over the time window T_0
T_0 = 10
num_window_steps = int(T_0 / dt)
avg_frequencies = np.mean(instantaneous_frequencies[:, :, -num_window_steps:], axis=-1)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot amplitude of the first oscillator at the final time
im1 = axs[0].imshow(amplitudes, cmap='viridis', origin='upper', extent=(0, lattice_size[1], lattice_size[0], 0))
axs[0].set_title('Amplitude of the First Oscillator at Final Time')
axs[0].set_xlabel('y')
axs[0].set_ylabel('x')
axs[0].grid(False)
cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical')
cbar1.set_label('Amplitude')

# Plot phase of the first oscillator at the final time
im2 = axs[1].imshow(phases_final, cmap='hsv', origin='upper', extent=(0, lattice_size[1], lattice_size[0], 0))
axs[1].set_title('Phase of the First Oscillator at Final Time')
axs[1].set_xlabel('y')
axs[1].set_ylabel('x')
axs[1].grid(False)
cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical')
cbar2.set_label('Phase (radians)')

# Set the ticks for the color bar to be in the range [0, 2π]
cbar2.set_ticks([0, np.pi, 2*np.pi])
cbar2.set_ticklabels(['0', '$\pi$', '$2\pi$'])

# Plot time-averaged frequency of the first oscillator
im3 = axs[2].imshow(avg_frequencies, cmap='viridis', origin='upper', extent=(0, lattice_size[1], lattice_size[0], 0))
axs[2].set_title('Time-Averaged Frequency of the First Oscillator')
axs[2].set_xlabel('y')
axs[2].set_ylabel('x')
axs[2].grid(False)
cbar3 = fig.colorbar(im3, ax=axs[2], orientation='vertical')
cbar3.set_label('Frequency (rad/s)')


# Visualize lattice graph
G = nx.grid_2d_graph(lattice_size[0], lattice_size[1])
pos = {(x, y): (y, -x) for x, y in G.nodes()}
plt.figure(figsize=(6, 6))
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
plt.title('Lattice Graph')
plt.show()

# Display the plot
plt.tight_layout()
plt.show()

