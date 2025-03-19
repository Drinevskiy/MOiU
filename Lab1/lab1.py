import numpy as np
from np_inverse_matrix import inverse_matrix_with_replaced_column
from custom_inverse_matrix import custom_inverse_matrix

# Функция для вывода результатов
def print_results(matrix, x, i):
    print("Матрица:\n", matrix, end="\n\n")
    # print("Обратная матрица:\n", np.linalg.inv(matrix), end="\n\n")

    # np_inverse = inverse_matrix_with_replaced_column(matrix, x, i - 1)
    custom_inverse = custom_inverse_matrix(matrix, x, i - 1)
    
    # print("Обратная матрица (с измененным вектором) numpy:\n", np_inverse, end="\n\n")
    print("Обратная матрица (с измененным вектором) custom:\n", custom_inverse, end="\n\n")

# Примеры
examples = [
    # (np.array([[1, 0, 0], 
    #            [0, 1, 0], 
    #            [0, 0, 1]]), np.array([6, 1, 0]), 2),
    (np.array([[1, 6, 0], 
               [0, 1, 0], 
               [0, 0, 1]]), np.array([9, 0, 1]), 1),
    # (np.array([[1, 0, 5], 
    #            [2, 1, 6], 
    #            [3, 4, 0]]), np.array([2, 2, 2]), 2),
    # (np.array([[1, -1, 0], 
    #            [0, 1, 0], 
    #            [0, 0, 1]]), np.array([1, 0, 1]), 3),
    # (np.array([[4, 2, 1], 
    #            [0, 3, 5], 
    #            [7, 1, 2]]), np.array([0, 0, 0]), 1),
    # (np.array([[2, 3, 1], 
    #            [4, 1, 5], 
    #            [6, 2, 0]]), np.array([0, 1, 2]), 3),
    # (np.array([[5, 1, 2], 
    #            [3, 4, 6], 
    #            [1, 0, 7]]), np.array([3, 3, 3]), 3),
    # (np.array([[1, 2, 3], 
    #            [0, 1, 4], 
    #            [5, 6, 0]]), np.array([7, 8, 9]), 1),
    # (np.array([[3, 2, 1], 
    #            [1, 4, 3], 
    #            [2, 1, 5]]), np.array([1, 1, 1]), 2),
    # (np.random.rand(10, 10), np.random.rand(10), 1),
    # (np.random.rand(10, 10), np.random.rand(10), 5),
    # (np.random.rand(10, 10), np.random.rand(10), 7),
]

# Проходим по всем примерам
for matrix, x, i in examples:
    print_results(matrix, x, i)