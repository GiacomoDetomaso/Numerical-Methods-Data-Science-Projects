from distutils.log import error
import numpy as np


def _swap_matrix_row(M, index_row1, index_row2):
    '''
        Swap the row of a matrix M.
        Params: 
        - M: input matrix
        - index_row1: row index of the first element 
        - index_row2: row index of the second element
    '''
    M[[index_row1, index_row2], :] = M[[index_row2, index_row1], :]


def _swap_matrix_column(M, index_col1, index_col2):
    '''
        Swap the column of a matrix M.
        Params: 
        - M: input matrix
        - index_col1: column index of the first element
        - index_col2: column index of the second element
    '''
    M[:, [index_col1, index_col2]] = M[:, [index_col2, index_col1]]


def _gauss_step(A, b, d_pivot_index, row_size, col_size):
    '''
        Executes the standard gauss step.
        Params: 
        - A: matrix
        - b: known terms
        - d_pivot_index: index of the pivot found on the matrix's main diagonal
        - row_size: number of rows
        - col_size: number of columns
    '''
    for i in range(d_pivot_index + 1, row_size):

        m = A[i][d_pivot_index] / A[d_pivot_index][d_pivot_index]
        A[i][d_pivot_index] = 0

        for k in range(d_pivot_index + 1, col_size):
            A[i][k] = A[i][k] - np.dot(m, A[d_pivot_index][k])

        b[i] = b[i] - np.dot(m, b[d_pivot_index])


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
    pivot = abs(A[0][0])
    row_pivot = 0
    col_pivot = 0

    # scan the matrix
    for i in range(dim):
        for j in range(dim):
            if abs(A[i][j]) > pivot:
                pivot = abs(A[i][j])
                row_pivot = i
                col_pivot = j

    return row_pivot, col_pivot


def _restore_matrix(A, b, index_rows, index_columns):
    '''
        This function is used in gauss with total pivoting 
        to exchange rows and columns after Gauss elimination is completed. 
        INPUT:
        - A: upper triangular matrix
        - b: known terms vector
        - index_rows: indeces of the rows to swap
        - index_columns: indeces of the columns to swap
    '''
    for i in range(len(index_rows)):
        (row1, row2) = index_rows[i]
        _swap_matrix_row(A, row1, row2)
        _swap_matrix_row(b, row1, row2)

    for j in range(len(index_columns)):
        (col1, col2) = index_columns[j]
        _swap_matrix_column(A, col1, col2)


def gauss_pivoting_partial(A, b):
    '''
        Performs gauss algorithm on a matrix A
        using the max partial pivot as technique

        INPUT: 
        - A: matrix to perform gauss elimination
        - b: known terms vectors 
    '''
    if np.linalg.det(A) == 0:
        error("Det = 0: singular matrix, cannot apply gauss ")
        return

    (row, col) = np.shape(A)  # get the shap of A

    for j in range(0, col-1):
        # set the first element of the column as the max pivot
        amax = abs(A[j][j])
        imax = j  # column index

        # scan all the row of the j column of the matrix in order to find the max partial pivot, starting from the j+1 row
        for i in range(j+1, row):
            if abs(A[i][j]) > amax:
                amax = abs(A[i][j])
                imax = i

        # row swap step
        if imax > j:
            _swap_matrix_row(A, j, imax)  # swap A rows
            _swap_matrix_row(b, j, imax)  # swap vector b rows

        # execute gauss elimination standard algorithm
        _gauss_step(A, b, j, row, col)


def gauss_pivoting_total(A, b):
    '''
        Performs gauss algorithm on a matrix A
        using a complete pivot technique

        INPUT: 
        - A: matrix to perform gauss elimination
        - b: known terms vectors 
    '''
    (row, col) = np.shape(A)  # get the shap of A

    exchanged_rows = []
    exchanged_columns = []

    (row_index_pivot, col_index_pivot) = _get_total_pivot(A, row)

    for j in range(0, col - 1):
        # row swap step
        if row_index_pivot > j or col_index_pivot > j:
            _swap_matrix_row(A, j, row_index_pivot)  # swap A rows
            _swap_matrix_row(b, j, row_index_pivot)  # swap vector b rows
            _swap_matrix_column(A, j, col_index_pivot)  # swap the column

            exchanged_rows.append((j, row_index_pivot))
            exchanged_columns.append((j, col_index_pivot))

        # execute gauss elimination standard algorithm
        _gauss_step(A, b, j, row, col)

        (m_s, n_s) = np.shape(A[j+1: col, j+1: col])
        (row_index_pivot, col_index_pivot) = _get_total_pivot(
            A[j+1: col, j+1: col], m_s)

    _restore_matrix(A, b, exchanged_rows, exchanged_columns)
