from copy import deepcopy
import numpy as np


def place_marks_on_basis(B, basis):
    mark = not B[basis]
    for i, j in B.keys():
        if basis[0] == i and B[(i, j)] is None:
            B[(i, j)] = mark
            place_marks_on_basis(B, (i, j))
        if basis[1] == j and B[(i, j)] is None:
            B[(i, j)] = mark
            place_marks_on_basis(B, (i, j))


def potentials_method(a, b, c):
    m = len(a)
    n = len(b)
    x = np.zeros((m, n)) 
    B = []  # Базисные позициии

    #! 1 Заполнение матрицы X(начальный базисный допустимый план)
    i = 0
    j = 0
    while i < m and j < n:
        amount = min(a[i], b[j])
        x[i, j] = amount
        B.append((i, j))
        a[i] -= amount
        b[j] -= amount
        if a[i] == 0 and i < len(a) - 1:
            i += 1
        elif b[j] == 0:
            j += 1

    #! 2 Основная фаза
    while True:
        #! 3 Нахождение u, v
        u_v_matrix = []
        result_values = []
        for i, j in B:
            row = [0] * (m + n)
            row[i] = 1  # u[i]
            row[m + j] = 1  # v[j]
            u_v_matrix.append(row)
            result_values.append(c[i, j])

        # 4 Фиксируем значение для u[0] = 0
        initial_condition = [0] * (m + n)
        initial_condition[0] = 1
        u_v_matrix.append(initial_condition)
        result_values.append(0)

        potentials = np.linalg.solve(u_v_matrix, result_values)
        u = potentials[: m]
        v = potentials[m :]

        #! 5 Проверка оптимальности
        new_basis_position = None
        for i in range(len(c)):
            for j in range(len(c[0])):
                # Проверка неравнества u[i] + v[j] > c[i,j], тогда нужно обновить план
                if (i, j) not in B and u[i] + v[j] > c[i, j]:
                    new_basis_position = (i, j)
                    break
            if new_basis_position:
                break

        if new_basis_position is None:
            return x

        B.append(new_basis_position)
        B = sorted(B)
        tmp_B = deepcopy(B)

        # Удаление базисных позиций в строках, в которых не более 1 базисной позиции
        for i in range(len(x)):
            counter = 0
            for j in range(len(x)):
                if (i, j) in tmp_B:
                    counter += 1
            if counter <= 1:
                for j in range(len(x)):
                    if (i, j) in tmp_B:
                        tmp_B.remove((i, j))

        # Удаление базисных позиций в столбцах, в которых не более 1 базисной позиции
        for j in range(len(x)):
            counter = 0
            for i in range(len(x)):
                if (i, j) in tmp_B:
                    counter += 1
            if counter <= 1:
                for i in range(len(x)):
                    if (i, j) in tmp_B:
                        tmp_B.remove((i, j))

        # 6 проставление элементам значений '+', '-'
        basis_marks = {item: None for item in tmp_B}
        basis_marks[new_basis_position] = True
        place_marks_on_basis(basis_marks, new_basis_position)

        # 7 Нахождение минимального значения из позиций с '-'
        min_value = np.inf
        for i in range(len(x)):
            for j in range(len(x)):
                if not basis_marks.get((i, j), True):
                    min_value = min(min_value, x[i, j])

        #! 8 Добавление min_value к позициям с '+', вычитание из позиций с '-'
        for (i, j) in basis_marks.keys():
            if basis_marks[(i, j)]:
                x[i, j] += min_value
            else:
                x[i, j] -= min_value

        #! 9 Удаление элементов с 0 значением
        for i, j in B:
            if x[i, j] == 0 and not basis_marks.get((i, j), True):
                B.remove((i, j))
                break


def main():
    examples = [
        (np.array([100, 300, 300]), # поставщики
        np.array([300, 200, 200]), # потребители
        np.array(
            [
                [8, 4, 1],
                [8, 4, 3],
                [9, 7, 5],
            ]
        )), # матрица стоимости
    ]

    for a, b, c in examples:
        print(potentials_method(a, b, c))


if __name__ == "__main__":
    main()
