# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:10:33 2024

@author: snagchowdh
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx

# Define the function representing the coupled system of ODEs
def z_dot(t, z, alpha, beta, omega, c, A):
    N = len(z)  # Number of nodes
    dzdt = np.empty_like(z, dtype=np.complex128)
    for j in range(N):
        dzdt[j] = (1j*omega[j] + alpha - beta*(abs(z[j])**2)) * z[j]
        for k in range(N):
            dzdt[j] += c * A[j, k] * (z[k] - z[j])
    return dzdt

# Define the size of the lattice
N = 3
# Define the number of nodes
num_nodes = N**2

# Define the parameters
alpha = 1.0
beta = 1
omega = np.array([3.0] * num_nodes)  # Frequencies for each node
c = 1.0

# Define the adjacency matrix for a 2D lattice with N=3
A = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    if i % N != N - 1:  # Check if not in the last column
        A[i, i + 1] = 1  # Connect to the right neighbor
    if i % N != 0:  # Check if not in the first column
        A[i, i - 1] = 1  # Connect to the left neighbor
    if i < num_nodes - N:  # Check if not in the last row
        A[i, i + N] = 1  # Connect to the bottom neighbor
    if i >= N:  # Check if not in the first row
        A[i, i - N] = 1  # Connect to the top neighbor

# # Plot the adjacency matrix
# plt.figure(figsize=(5, 5))
# plt.imshow(A, cmap='binary', interpolation='nearest')
# plt.title('Adjacency Matrix (2D Lattice)')
# plt.colorbar()
# plt.show()

# # Create the graph from the adjacency matrix
# G = nx.Graph(A)

# # Visualize the graph
# plt.figure(figsize=(5, 5))
# nx.draw(G, with_labels=True)
# plt.title('Graph Visualization')
# plt.show()

# Define the time points for integration
t_span = (0, 2000)  # Integration time interval
step_length = 0.01
t_eval = np.arange(t_span[0], t_span[1], step_length)

# Define the function to generate random initial conditions
def random_initial_conditions(N):
    return [complex(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in range(N)]

# Solve the differential equation for random initial conditions
initial_conditions = random_initial_conditions(num_nodes)
solution = solve_ivp(z_dot, t_span, initial_conditions, args=(alpha, beta, omega, c, A), t_eval=t_eval, method='RK45')

# Extract the solution
z_solution = solution.y
t_solution = solution.t

# Plot real parts of z for the final iterations
plt.figure(figsize=(10, 5))
for j in range(len(omega)):
    plt.plot(t_solution[-1000:], np.real(z_solution[j, -1000:]), label=f'Real(z_{j+1})')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Real Parts of z_j(t) for Final Iterations')
plt.legend()
plt.grid(True)
plt.show()


######################################################

# Extract the solution
z_solution = solution.y
t_solution = solution.t

# Extract the amplitudes at the last time step
final_amplitudes = np.abs(z_solution[:, -1])

# Create the lattice graph
G = nx.grid_2d_graph(N, N)

# Draw the lattice graph with node colors representing the amplitudes
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)  # Position nodes using a spring layout

# Plot nodes with colors corresponding to the amplitudes
nodes = nx.draw_networkx_nodes(G, pos, node_color=final_amplitudes, cmap='viridis', vmin=0, vmax=1)
nx.draw_networkx_edges(G, pos, alpha=0.5)  # Draw edges
plt.title('Amplitude of Oscillators at Last Time Step')

# Create colorbar using the mappable object
cbar = plt.colorbar(nodes, label='Amplitude')
plt.show()


#########################################################################

# Extract the phases at the last time step
final_phases = np.angle(z_solution[:, -1])


# Draw the lattice graph with node colors representing the phases
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)  # Position nodes using a spring layout

# Plot nodes with colors corresponding to the phases
nodes = nx.draw_networkx_nodes(G, pos, node_color=final_phases, cmap='hsv', vmin=-np.pi, vmax=np.pi)
nx.draw_networkx_edges(G, pos, alpha=0.5)  # Draw edges
plt.title('Phase of Oscillators at Last Time Step')

# Create colorbar using the mappable object
cbar = plt.colorbar(nodes, label='Phase')
cbar.set_ticks([-np.pi, 0, np.pi])
cbar.set_ticklabels(['$-\pi$', '0', '$\pi$'])

plt.show()