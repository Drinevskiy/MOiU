import numpy as np

def inverse_matrix_with_replaced_column(matrix, x, i):
    # Проверка на соответствие размеров
    if matrix.shape[0] != len(x):
        raise ValueError("Размеры вектора x должны соответствовать количеству строк матрицы.")
    
    # Создание копии матрицы
    modified_matrix = matrix.copy()
    
    # Замена i-ого столбца на вектор x
    modified_matrix[:, i] = x
    
    # Нахождение обратной матрицы
    try:
        inv_matrix = np.linalg.inv(modified_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Матрица не является обратимой.")
    
    return inv_matrix