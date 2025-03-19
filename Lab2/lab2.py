import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lab1.custom_inverse_matrix import custom_inverse_matrix

import numpy as np

def simplex_method(c, A, x, B, firstIter = True, j0 = -1, i = -1):
    # 1
    if firstIter:
        A_B = A[:, B]
        A_B_inverse = np.linalg.inv(A_B)
    else:
        A_B = A[:, B]
        A_B_inverse = custom_inverse_matrix(A_B, A[:, j0], i)
    # 2
    c_B = c[B]
    # 3 вектор потенциалов
    U = c_B @ A_B_inverse
    # 4 вектор оценок
    delta = U @ A - c
    # 5 - 6
    j0 = -1
    for index, value in enumerate(delta):
        if value < 0:
            j0 = index
            break
    if j0 == -1:
        return x, B
    # 7
    z = A_B_inverse @ A[:,j0]
    # 8
    tetta = []
    for index, value in enumerate(z):
        if value > 0:
            tetta_i = x[B[index]] / value  # Индекс B[i] соответствует базисному индексу
        else:
            tetta_i = np.inf
        tetta.append(tetta_i)
    # 9
    tetta0 = min(tetta)
    # 10
    if tetta0 == np.inf:
        return "Целевая функция не ограничена сверху на множестве допустимых планов"
    # 11
    tetta0_index = tetta.index(tetta0)
    j_star = B[tetta0_index]
    # 12
    B[tetta0_index] = j0
    # 13
    x[j0] = tetta0
    for index, value in enumerate(z):
        if tetta0_index != index:
            x[B[index]] = x[B[index]] - tetta0 * value
    x[j_star] = 0
    return simplex_method(c, A, x, B, False, j0, tetta0_index)

examples = [
    (np.array([1, 1, 0, 0, 0], dtype=float),
     np.array([[-1, 1, 1, 0, 0], 
               [1, 0, 0, 1, 0], 
               [0, 1, 0, 0, 1]], dtype=float), 
     np.array([0, 0, 1, 3, 2], dtype=float),
     np.array([2,3,4])),

    (np.array([10, 13, 0, 0, 0], dtype=float),
     np.array([[6, 9, 1, 0, 0], 
               [1, 0, 0, 1, 0], 
               [0, 1, 0, 0, 1]], dtype=float), 
     np.array([0, 0, 200, 30, 20], dtype=float),
     np.array([2,3,4])),


    (np.array([1, 0], dtype=float),
     np.array([[1, -1]], dtype=float), 
     np.array([1, 0], dtype=float),
     np.array([0])),
]

# def print_results(c, A, b):
#     x = b
#     B = [i for i in range(len(x)) if x[i] != 0]
#     print(simplex_method(c, A, x, B))

# for c, A, x, B in examples:
#     print(simplex_method(c, A, x, B))
    # print_results(c, A, )
