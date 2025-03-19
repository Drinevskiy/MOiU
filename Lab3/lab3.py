import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lab2.lab2 import simplex_method

# def correct_basis_index():

def start_simplex_method(c, A, b):
    n = len(c)
    m = len(b)
    # 1
    for index, bi in enumerate(b):
        if bi < 0:
            b[index] = -1 * bi
            A[index, :] = -1 * A[index, :]
    # 2
    c_wave = np.concatenate((np.zeros(n), -1 * np.ones(m)))
    A_wave = np.hstack((A, np.eye(m)))
    # 3
    x_wave = np.concatenate((np.zeros(n), b))
    B_wave = [n + i for i in range(m)]
    # 4
    # print(c_wave)
    # print(A_wave)
    # print(x_wave)
    # print(B_wave)

    result = simplex_method(c_wave, A_wave, x_wave, B_wave)
    # Проверка, является ли возвращаемое значение строкой (что указывает на ошибку)
    if isinstance(result, str):
        print(result) 
    else:
        x_wave, B = result 
    # 5
    for i in range(n, n + m):
        if x_wave[i] != 0:
            return "Задача несовместна."
    # 6
    x = x_wave[:n]
    # 7
    while True:
        if all(bi < n for bi in B):
            return x, B, A, b
    # 8    
        j_k = max(B)
        k = B.index(j_k) 
    # 9
        j_nb = [i for i in range(n) if all(bi != i for bi in B)] 
        A_wave_B = A_wave[:,B]
        A_wave_B_inv = np.linalg.inv(A_wave_B)
        l = [(j, A_wave_B_inv @ A_wave[:, j]) for j in j_nb]
    # 10
        found = False 
        for j, l_j in l:
            if(l_j[k] != 0):
                B[k] = j 
                found = True
        if not found:
            index = j_k - n
            A = np.delete(A, index, axis=0)
            b = np.delete(b, index)
            B = np.delete(B, k)
            A_wave = np.delete(A_wave, index, axis=0)

examples = [
    (np.array([1, 0, 0], dtype=float),
     np.array([[1, 1, 1], 
               [2, 2, 2]], dtype=float), 
     np.array([0, 0], dtype=float)),

    # (np.array([1, 1,], dtype=float),
    #  np.array([[-1, 1], 
    #            [1, 0,], 
    #            [0, 1,]], dtype=float), 
    #  np.array([1, 3, 2], dtype=float)),

    # (np.array([10, 13,], dtype=float),
    #  np.array([[6, 9,], 
    #            [1, 0,], 
    #            [0, 1,]], dtype=float), 
    #  np.array([200, 30, 20], dtype=float)),

    # (np.array([1, 0], dtype=float),
    #  np.array([[1, -1]], dtype=float), 
    #  np.array([1], dtype=float)),
]

for c, A, b in examples:
    x, B, A_new, b_new = start_simplex_method(c, A, b)
    print(simplex_method(c, A_new, x, B))
    # print(B)
    # print(A_new)
    # print(b_new)