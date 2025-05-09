import numpy as np

def square_method(c, D, A, x, B, B_ext):
    # 1 Находим delta_x
    c_x = c + D @ x
    u_x = -c_x[B] @ np.linalg.inv(A[:, B])
    delta_x = u_x @ A + c_x

    # if not np.all(delta_x[B_ext] == 0):
    #     return "В векторе delta не все элементы с индексами из B_ext равны 0. Задача не имеет решений."

    # 2-3 Проверка оптимальности
    j0 = -1
    for index, value in enumerate(delta_x):
        if value < 0:
            j0 = index
            break
    if j0 == -1:
        return x
    
    # 4 Нахождение вектора l
    B_not_in_ext = [i for i in range(len(A[0])) if i not in B]
    l_B_not_in_ext = [((i, 1) if i == j0 else (i, 0)) for i in B_not_in_ext]

    D_star = D[B][:, B]
    A_star = A[:, B]
    M_zero = np.zeros((len(B), len(B)))

    H = np.block([
        [D_star, np.linalg.inv(A_star)],
        [A_star, M_zero]
    ])

    b_top = D[:, j0][B]
    b_bottom = A[:, j0]
    b_star = np.block([b_top, b_bottom])

    x_hat = -1 * np.linalg.inv(H) @ b_star

    count = 0
    l_B_ext = []
    for j in B:
        l_B_ext.append((int(j), float(x_hat[count])))
        count = count + 1

    value_dict = {idx: val for idx, val in l_B_ext + l_B_not_in_ext}
    l = [value_dict[i] for i in sorted(value_dict.keys())]

    # 5 Нахождение tetta0
    delta = l @ D @ l
    tetta_j0 = np.inf if delta == 0 else abs(delta_x[j0]) / delta

    tetta = [(int(j), float(-x[j]/l[j])) if l[j] < 0 else (int(j), np.inf) for j in B_ext]
    tetta.append((j0, tetta_j0))
    (j_star, tetta0) = min(tetta, key=lambda x: x[1])
    if tetta0 == np.inf:
        return "Целевая функция не ограничена снизу на множестве допустимых планов"

    # 6 Преобразование плана и индексов
    x = x + tetta0 * np.array(l)
    # print(x)
    B_ext_minus_B = [j for j in B_ext if j not in B]
    if j_star == j0:
        B_ext = np.append(B_ext, j_star)
        return square_method(c, D, A, x, B, B_ext)
        # B_ext.append(j_star)
    elif j_star in B_ext_minus_B:
        # B_ext.remove(j_star)
        B_ext = np.delete(B_ext, np.where(B_ext == j_star))
        return square_method(c, D, A, x, B, B_ext)
    
    s = np.where(B == j_star)[0][0]
    # s = B.index(j_star)
    A_B_inverse = np.linalg.inv(A[:, B])
    j_plus = -1

    for j in B_ext_minus_B:
        if (A_B_inverse @ A[:, j])[s] != 0:
            j_plus = j
            break
    if j_star in B and j_plus != -1:
        B_ext = np.delete(B_ext, np.where(B_ext == j_star))
        B[np.where(B == j_star)[0]] = j_plus
        # B_ext.remove(j_star)
        # B[B.index(j_star)] = j_plus 
        return square_method(c, D, A, x, B, B_ext)
    
    if B == B_ext or all((A_B_inverse @ A[:, j])[s] == 0 for j in B_ext_minus_B):
        B[np.where(B == j_star)[0]] = j0
        B_ext[np.where(B_ext == j_star)[0]] = j0
        # B[B.index(j_star)] = j0
        # B_ext[B_ext.index(j_star)] = j0
        return square_method(c, D, A, x, B, B_ext)

examples = [
    (np.array([-8, -6, -4, -6], dtype=float), # c
     np.array([[2, 1, 1, 0], 
               [1, 1, 0, 0],
               [1, 0, 1, 0],
               [0, 0, 0, 0],], dtype=float), # D
     np.array([[1, 0, 2, 1],
              [0, 1, -1, 2]], dtype=float), # A
     np.array([2, 3, 0, 0], dtype=float), # x
     np.array([0, 1]), # B
     np.array([0, 1])), # B_ext
]

for c, D, A, x, B, B_ext in examples:
    result = square_method(c, D, A, x, B, B_ext)
    print(result)
    # if isinstance(result, str):
    #     print(result) 
    # else:
    #     cappa = result 
    #     print(cappa)