# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:28:09 2024

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
N = 10  # Number of systems
alpha = np.ones(N)  # All systems have same alpha value
beta = np.ones(N)   # All systems have same beta value
omega = 3.0  # All systems have same omega value
k = 1.0  # Coupling strength

# Adjacency matrix for a square lattice
A = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j and abs(i - j) == 1:
            A[i, j] = 1
            A[j, i] = 1

# Initial conditions for x and y for each system
init_conditions = np.random.uniform(-1, 1, size=(2*N))

# Solve the coupled systems using RKF45 method
t_span = (0, 2000)
t_eval = np.linspace(*t_span, 200000)

sol = solve_ivp(coupled_system, t_span, init_conditions, args=(alpha, beta, omega, A, k, N), t_eval=t_eval, method='RK45')

# Plot phase portrait for one system
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(sol.t[-1000:], sol.y[0][-1000:], linewidth=0.5)
plt.title('$t$ vs $x$ - One Coupled System')
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.grid(True)

# Plot square lattice
plt.subplot(1, 2, 2)
G = nx.Graph()
G.add_edges_from([(i, j) for i in range(N) for j in range(N) if i != j and abs(i - j) == 1])
pos = nx.spring_layout(G)  # Positions for all nodes
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', linewidths=1, font_size=10)
plt.title('Square Lattice')
plt.show()
