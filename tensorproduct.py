# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:45:48 2024

@author: snagchowdh
"""

import numpy as np

def kronecker_delta(i, j):
    return 1 if i == j else 0

# Pauli X matrix
sigma_x = np.array([[0, 1],
                     [1, 0]])

# Pauli Y matrix
sigma_y = np.array([[0, -1j],
                     [1j, 0]])

# Pauli Z matrix
sigma_z = np.array([[1, 0],
                     [0, -1]])

# Create an identity matrix of order 2
I = np.eye(2)




# print("Pauli X matrix:")
# print(sigma_x)
# print()

# print("Pauli Y matrix:")
# print(sigma_y)
# print()

# print("Pauli Z matrix:")
# print(sigma_z)


# Compute the tensor product of A and B
tensor_product = np.kron(I, sigma_y)

print("Tensor product of A and B:")
print(tensor_product)



