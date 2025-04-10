import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lab1.custom_inverse_matrix import custom_inverse_matrix

import numpy as np

def double_simplex_method(c, A, b, B, firstIter = True, j0 = -1, i = -1):
    n = len(c)
    # 1
    if firstIter:
        A_B = A[:, B]
        A_B_inverse = np.linalg.inv(A_B)
    else:
        A_B = A[:, B]
        A_B_inverse = custom_inverse_matrix(A_B, A[:, j0], i)
    # 2
    c_B = c[B]
    # 3
    y = c_B @ A_B_inverse
    # 4
    cappa_B = A_B_inverse @ b
    cappa = np.zeros(n)
    cappa[B] = cappa_B
    # 5
    if np.all(cappa >= 0):
        return cappa
    # 6
    j_k = np.where(cappa < 0)[0][-1]
    k = np.where(B == j_k)[0][0]
    # 7
    delta_y = A_B_inverse[k, :]
    j_nb = [i for i in range(n) if all(bi != i for bi in B)] 
    mu = [(j, delta_y @ A[:, j]) for j in j_nb]
    # 8
    if all(mu_j >= 0 for j, mu_j in mu):
        return "Прямая задача не совместна"
    # 9
    sigma = [(c[j] - A[:, j] @ y) / mu_j for j, mu_j in mu if mu_j < 0]
    # 10
    sigma0 = min(sigma)
    sigma0_index = sigma.index(sigma0)
    # 11
    B[k] = sigma0_index
    return double_simplex_method(c, A, b, B, False,  sigma0_index, k)

examples = [
    (np.array([-4, -3, -7, 0, 0], dtype=float),
     np.array([[-2, -1, -4, 1, 0], 
               [-2, -2, -2, 0, 1]], dtype=float), 
     np.array([-1, -1.5], dtype=float),
     np.array([3, 4])),
]

for c, A, b, B in examples:
    result = double_simplex_method(c, A, b, B)
    if isinstance(result, str):
        print(result) 
    else:
        cappa = result 
        print(cappa)
    
    