from turtle import shape
import numpy as np


def matrix_product(A, B):
    '''
        This function performs the product of a 
        matrix A and a matrix B.
        INPUT:
        - A: matrix 
        - B: matrix
        OUTPUT:
        - C: product matrix
    '''
    (rowsA, colsA) = np.shape(A)
    (rowsB, colsB) = np.shape(B)

    if colsA == rowsB:
        C = np.zeros((rowsA, colsB))  # result matrix

        for i in range(rowsA):
            for j in range(colsB):
                for k in range(colsA):
                    C[i][j] = C[i][j] + np.dot(A[i][k], B[k][j])

        return C
    else:
        return 0


def lt_linear_system_solver(L, b):
    '''
        This function is used to solve a linear system
        with a lower triangular matrix as input.
        INPUT: 
        - L: lower triangular matrix
        - b: array with the known terms
        OUTPUT: 
        - (solved, x): solved is a flag that indicates if the system has been solved while x are the solutions 
    '''
    det_L = np.linalg.det(L)
    solved = True

    # result vector x initialization
    (m, n) = np.shape(b)  # m is the number of rows and n the number of column
    x = np.zeros((m, n))

    if det_L == 0:
        solved = False
        return solved

    x[0] = b[0] / L[0][0]  # x1 calculation

    for k in range(1, m):
        x[k] = (b[k] - np.dot(L[k][0:k], x[0:k])) / L[k][k]

    return solved, x


def ut_linear_system_solver(U, b):
    '''
        This function is used to solve a linear system
        with an upper triangular matrix as input

        INPUT: 
        - U: upper triangular matrix
        - b: array with the known terms
        OUTPUT: 
        - (solved, x): solved is a flag that indicates if the system has been solved while x are the solutions 
    '''
    det_L = np.linalg.det(U)
    solved = True

    # result vector x initialization
    (m, n) = np.shape(b)  # m is the number of rows and n the number of column
    x = np.zeros((m, n))

    if det_L == 0:
        solved = False
        return solved

    x[m - 1] = b[m - 1] / U[m - 1][m - 1]  # xn calculation

    for k in reversed(range(m - 1)):
        x[k] = (b[k] - np.dot(U[k][k+1:m], x[k+1:m])) / U[k][k]

    return solved, x





