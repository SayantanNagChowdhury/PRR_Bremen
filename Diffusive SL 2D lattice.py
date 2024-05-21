# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:40:57 2024

@author: snagchowdh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import networkx as nx

# Define the system of differential equations for coupled systems
def coupled_system(t, variables, alpha, beta, omega, A, k, N):
    x = variables[:N]
    y = variables[N:]

    dxdt = np.zeros(N)
    dydt = np.zeros(N)

    for i in range(N):
        dxdt[i] = (alpha[i] - beta[i] * (x[i]**2 + y[i]**2)) * x[i] - omega * y[i]
        dydt[i] = (alpha[i] - beta[i] * (x[i]**2 + y[i]**2)) * y[i] + omega * x[i]

        for j in range(N):
            dxdt[i] += k * A[i, j] * (x[j] - x[i])
            dydt[i] += k * A[i, j] * (y[j] - y[i])

    return np.concatenate([dxdt, dydt])

# Parameters
N = 3  # Size of the square lattice (N x N)
alpha = np.ones(N * N)  # All systems have same alpha value
beta = np.ones(N * N)   # All systems have same beta value
omega = 3.0  # All systems have same omega value
k = 1.0  # Coupling strength

# Construct adjacency matrix for a 2D square lattice
A = np.zeros((N * N, N * N))
for i in range(N):
    for j in range(N):
        idx = i * N + j 
        if i > 0:
            A[idx, idx - N] = 1 # Connect to the node above
        if i < N - 1:
            A[idx, idx + N] = 1 # Connect to the node below
        if j > 0:
            A[idx, idx - 1] = 1 # Connect to the node to the left
        if j < N - 1:
            A[idx, idx + 1] = 1 # Connect to the node to the right

# Initial conditions for x and y for each system
init_conditions = np.random.uniform(-1, 1, size=(2 * N * N))

# Solve the coupled systems using RKF45 method
t_span = (0, 2000)
t_eval = np.linspace(*t_span, 200000)

sol = solve_ivp(coupled_system, t_span, init_conditions, args=(alpha, beta, omega, A, k, N * N), t_eval=t_eval, method='RK45')

# Plot phase portrait for one system
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)


#plt.figure(figsize=(12, 6))
for i in range(N * N):
    plt.plot(sol.t[-1000:], sol.y[i][-1000:], linewidth=0.5, label=f'$x_{i+1}$')

plt.title('$t$ vs $x_i$ for All Oscillators')
plt.xlabel('$t$')
plt.ylabel('$x_i$')
plt.legend()
plt.grid(True)
#plt.show()


# Plot square lattice
plt.subplot(1, 2, 2)
G = nx.grid_2d_graph(N, N)
pos = {(x, y): (y, -x) for x, y in G.nodes()}  # Rotate nodes to have origin at bottom-left
nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=500, edge_color='gray', linewidths=1)
plt.title('Square Lattice')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Plot all oscillators x vs y
plt.figure(figsize=(8, 8))
for i in range(N * N):
    plt.plot(sol.y[i][-1000:], sol.y[i + N*N][-1000:], label=f'Oscillator {i+1}', linewidth=0.5)

plt.title('Phase Portrait - All Oscillators')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

