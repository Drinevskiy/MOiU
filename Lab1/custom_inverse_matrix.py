import numpy as np

def multiplicate_matrix(q_matrix, a_reverse_matrix, index):
    q_rows, q_cols = q_matrix.shape
    result_matrix = np.zeros((q_rows, q_cols))
    for i in range(q_rows):
        for j in range(q_cols):
            # Диагональный элемент Q матрицы
            result_matrix[i, j] +=  q_matrix[i, i] * a_reverse_matrix[i, j]
            if i != index:
                # i-ый элемент Q матрицы
                result_matrix[i, j] +=  q_matrix[i, index] * a_reverse_matrix[index, j]

    return result_matrix
    
def custom_inverse_matrix(matrix, x, i):
    if matrix.shape[0] != len(x):
        raise ValueError("Размеры вектора x должны соответствовать количеству строк матрицы.")
    if np.linalg.det(matrix) == 0:
        raise np.linalg.LinAlgError("Определитель матрицы равен нулю; матрица не имеет обратной.")
    matrix = np.linalg.inv(matrix)
    l_vector = matrix @ x
    temp = l_vector[i]
    if temp == 0:
        raise ValueError("Матрица не обратима. l[i] = 0")
    l_vector[i] = -1
    l_vector_hat = -1 * l_vector / temp
    e_matrix = np.eye(len(x))
    q_matrix = e_matrix.copy()
    q_matrix[:, i] = l_vector_hat
    return multiplicate_matrix(q_matrix, matrix, i)