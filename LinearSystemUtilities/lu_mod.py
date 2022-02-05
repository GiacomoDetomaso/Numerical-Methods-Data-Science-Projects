from distutils.log import error
import numpy as np
import linear_system_mod as lsm


def _get_total_pivot(A, dim):
    '''
        This method is used to find the max total pivot on a matrix A
        INPUT: 
        - A: matrix
        - dim: dimension of the matrix
        OUTPU:
        - row_pivot: row index of the new pivot found
        - col_pivot: column index of the new pivot found
    '''
    pivot = abs(A[0, 0])
    row_pivot = 0
    col_pivot = 0

    # scan the matrix
    for i in range(dim):
        for j in range(dim):
            if abs(A[i, j]) > pivot:
                pivot = abs(A[i, j])
                row_pivot = i
                col_pivot = j

    return row_pivot, col_pivot


def Fatt_LU_No_piv(A):
    '''
        This function performes a LU factorization without pivoting .
        A = LU
        INPUT: 
        - A: matrix to factorized
        OUTPUT: 
        - L: lower triangular matrix
        - U: upper triangular matrix
    '''
    U = np.copy(A)
    n = U.shape[0]
    L = np.zeros((n, n))
    for j in range(n-1):
        L[j, j] = 1
        for i in range(j+1, n):
            L[i, j] = U[i, j]/U[j, j]
            U[i, j] = 0
            for k in range(j+1, n):
                U[i, k] = U[i, k] - L[i, j]*U[j, k]
    L[n-1, n-1] = 1

    return L, U

def LU_factorization_partial_pivoting(A):
    '''
        This function performes a LU factorization with a partial pivoting technique.
        PA = LU
        INPUT: 
        - A: matrix to factorized
        OUTPUT: 
        - P: permutation matrix of the row
        - L: lower triangular matrix
        - U: upper triangular matrix
    '''
    (a_m, a_n) = np.shape(A)  # misure matrice
    if a_m == a_n:
        n = a_m  # per semplificare la notazione

        U = np.copy(A)
        L = np.zeros([n, n])
        P = np.eye(n)

        if n == 1:
            print("Quello passato in input è un valore scalare")
            L = 1  # metto l'uno sulla diagonale di L
            return P, L, U
        else:
            p_index = 0 # actual pivot
            rmax = 0  # rows which cotains the new pivot to be changed  with the one in p_index position
            for i in range(n):
                rmax = np.argmax(abs(U[p_index:n, p_index]))
                rmax += p_index

                # pivoting
                U[[p_index, rmax], :] = U[[rmax, p_index], :]  # swap U rows
                L[[p_index, rmax], :] = L[[rmax, p_index], :]  # swap L rows
                P[[p_index, rmax], :] = P[[rmax, p_index], :]  # swap P rows

                # calculate submatrices  
                L[(p_index+1):n, p_index] = U[(p_index+1):n, p_index] / U[p_index, p_index]  # calculate L_21
                U[(p_index+1):n, (p_index+1):n] = U[(p_index+1):n, (p_index+1):n] - (L[(p_index+1):n, p_index] * U[p_index, (p_index+1):n])  # calculate the U_22 matrix
                U[(p_index+1):n, p_index] = 0 # turns into 0 every element under the pivot 

                p_index += 1 # point to the next pivot

            L += np.eye(n)  # insert the 1 on L diagonal
            return P, L, U

    else:
        print("Errore, la matrice A non è quadrata")


def LU_factorization_complete_pivoting(A):
    '''
        This function performes a LU factorization with a complete pivoting technique.
        PAQ = LU
        INPUT: 
        - A: matrix to factorized
        OUTPUT: 
        - P: permutation matrix of the row
        - Q: permutation matrix of the columns
        - L: lower triangular matrix
        - U: upper triangular matrix
    '''
    (a_m, a_n) = np.shape(A)

    if np.linalg.det(A) == 0:
        error("Det = 0 A is singular and cannot be factorized")
        return

    if a_m == a_n:
        n = a_m  # the matrix is nxn

        U = np.asmatrix(np.copy(A))
        L = np.asmatrix(np.zeros([n, n]))
        P = np.asmatrix(np.eye(n))
        Q = np.asmatrix(np.eye(n))

        if n == 1:
            print("Quello passato in input è un valore scalare")
            L = 1  # metto l'uno sulla diagonale di L
            return P, L, U, Q
        else:
            p_index = 0  # actual pivot index
            r = 0  # row of the max value of the considered matrix
            c = 0 # column of the max value of the considered matrix

            for i in range(1, n, 1):
                (m_s, n_s) = np.shape(U[p_index:n, p_index:n])
                (r, c) = _get_total_pivot(U[p_index:n, p_index:n], m_s)
                r += p_index
                c += p_index

                # pivoting
                P[[p_index, r], :] = P[[r, p_index], :]  # swap P rows
                U[[p_index, r], :] = U[[r, p_index], :]  # swap U rows
                L[[p_index, r], :] = L[[r, p_index], :]  # swap L rows
                Q[:, [p_index, c]] = Q[:, [c, p_index]]  # swap Q columns
                U[:, [p_index, c]] = U[:, [c, p_index]]  # swap U columns
                L[:, [p_index, c]] = L[:, [c, p_index]]  # swap L columns

                # calculate submatrices 
                L[(p_index+1):n, p_index] = U[(p_index+1):n, p_index] / U[p_index, p_index]  # calculate L_21
                U[(p_index+1):n, (p_index+1):n] = U[(p_index+1):n, (p_index+1):n] - (L[(p_index+1):n, p_index] * U[p_index, (p_index+1):n]) # calculate the U_22 matrix
                U[(p_index+1):n, p_index] = 0 # turns into 0 every element under the pivot  

                p_index += 1

            L += np.eye(n)  # insert the 1 on L diagonal

            return P, L, U, Q


def inverse_LU(P, A, L, U):
    '''
        This function performs the calculation of the 
        inverse of a matrix, knowing its LU factorization in the form: PA = LU
        INPUT: 
        - P: permutation matrix 
        - A: matrix to find the inverse
        - L: lower triangular matrix
        - U: upper triangular matrix
        OUTPUT: 
        - Y: intermediate inverse obtained by resolving the linear system LY = P
        - X: inverse of A obtained by resolving the linear system UX = Y
    '''
    Y = np.zeros(np.shape(A))
    X = np.zeros(np.shape(A))  # inverse
    (row, col) = np.shape(A)
    b = np.zeros((row, 1))

    # Y calculation: solve the linear equation LY = P
    for i in range(col):
        b = P[:, [i]]  # extract the column from P
        (sol, y) = lsm.lt_linear_system_solver(L, b)

        if sol is True:
            Y[:, [i]] = y
        else:
            error("An error occured!")
            return

    # X calculation: solve the linear equation UX = Y
    for i in range(col):
        b = Y[:, [i]]  # extract the column from Y
        (sol, x) = lsm.ut_linear_system_solver(U, b)

        if sol is True:
            X[:, [i]] = x
        else:
            error("An error occured!")
            return

    return Y, X


def inverse_Lu_compact(P, A, L, U):
    '''
        This function performs the calculation of the 
        inverse of a matrix, knowing its LU factorization in the form: PA = LU. 
        It is the compact version of inverse_LU function. The result it's just the inverse of A
        INPUT: 
        - P: permutation matrix 
        - A: matrix to find the inverse
        - L: lower triangular matrix
        - U: upper triangular matrix
        OUTPUT: 
        - X: inverse of A obtained by resolving the linear system UX = Y
    '''
    X = np.zeros(np.shape(A))
    (row, col) = np.shape(A)
    w = np.zeros((row, 1))  # corresponds to the column of P

    for i in range(col):
        w = P[:, [i]]
        LU = np.dot(L, U)
        x = np.linalg.solve(LU, w)
        X[:, [i]] = x

    return X
