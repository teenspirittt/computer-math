import numpy as np

SIZE = 4

matrix = [[9, 14, -15, 23],
          [16, 7, -22, 29],
          [18, 20, -3, 32],
          [10, 12, -16, 9]]

matrix2 = [[90, 14, -15, 23],
           [16, 70, -22, 29],
           [18, 20, -300, 32],
           [10, 12, -16, 90]]

b = [5, 8, 9, 4]


def determinant(matrix, m):
    p = [[0] * m for _ in range(m)]

    d = 0
    k = 1
    n = m - 1

    if m == 1:
        return matrix[0][0]

    if m == 2:
        return matrix[0][0] * matrix[1][1] - (matrix[1][0] * matrix[0][1])

    if m > 2:
        for i in range(m):
            getMatrix(matrix, p, i, 0, m)
            d = d + k * matrix[i][0] * determinant(p, n)
            k = -k
    return d


def getMatrix(matrix, p, i, j, m):
    di = 0
    for ki in range(m - 1):
        if ki == i:
            di = 1
        dj = 0
        for kj in range(m - 1):
            if kj == j:
                dj = 1
            p[ki][kj] = matrix[ki + di][kj + dj]


def swap_el(matrix, b, k, m):
    for i in range(m):
        matrix[i][k] = b[i]


def copy_matrix(matrix, m):
    tmp = [[m] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            tmp[i][j] = matrix[i][j]
    return tmp


def solution_print(sol, m):
    for i in range(m):
        print("x%d  = %.4f" % (i + 1, sol[i]))


def eMatrix(m):
    E = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                E[i][j] = 1
    return E


def LU_decomposition(L, U, m):
    for k in range(m - 1):
        for i in range(m - 1 - k):
            p = k + i + 1
            L[p][k] = U[p][k] / U[k][k]
            for j in range(m):
                U[p][j] = U[p][j] - U[k][j] * L[p][k]


def solY(L, y, b, m):
    for i in range(m):
        y[i] = b[i] - L[i][0] * y[0] - L[i][1] * y[1] - L[i][2] * y[2]


def solx(U, x, y, m):
    for i in range(m):
        k = m - 1 - i
        x[k] = (y[k] - U[k][3] * x[3] - U[k][2] * x[2] - U[k][1] * x[1]) / U[k][k]


def Kramer(matrix, b, m):
    sol = m * [0]
    d = determinant(matrix, m)
    dArray = [0] * m
    for i in range(m):
        dArray[i] = copy_matrix(matrix, m)
        swap_el(dArray[i], b, i, m)
        sol[i] = determinant(dArray[i], m) / d

    return sol


def CholeskyMethod(A, b, m):
    L = eMatrix(m)
    U = copy_matrix(A, m)
    LU_decomposition(L, U, m)
    y = [0] * SIZE
    x = [0] * SIZE
    solY(L, y, b, m)
    solx(U, x, y, m)
    return x


def ReverseMatrixMethod(matrix, b, SIZE):
    OM = np.linalg.inv(matrix)
    X2 = np.matmul(OM, b)
    return X2


print("Kramer system solution:")
solution_print(Kramer(matrix2, b, SIZE), SIZE)
print("")

print("Cholesky system solution")
solution_print(CholeskyMethod(matrix2, b, SIZE), SIZE)
print("")

print("Solution of the system by the inverse matrix method")
solution_print(ReverseMatrixMethod(matrix2, b, SIZE), SIZE)
print("\nTest:")
print("[ 0.03177144  0.09040831 -0.01935472  0.025419 ]")
