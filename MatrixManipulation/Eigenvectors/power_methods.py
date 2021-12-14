import numpy as np


def power_method_std(u0, A, tol, n_it_max):
    '''
        Standard implementation of the power method used to find 
        an approximation of the eigen values. 

        ### INPUT
            - u0:start vector
            - A: matrix whose eigenvector will be calculated
            - t: tolerance 
            - n_it_max: this is the max number of iterations

        ### OUTPUT
            - uk: eigenvector
            - actual_it: actual number of iteration before the stop
            - errors_list: a list that cotains the error of two consecutive eigen value lambda approximation
            - lamba_approx_list: list of approximantions of the eigen values
    '''
    acuatl_it = 0  # actual number of iterations

    uk = np.dot(A, u0)
    lam1 = np.dot(uk.T, np.dot(A, uk)) / (np.dot(uk.T, uk))

    lambda_approx_list = []
    errors_list = []

    lambda_approx_list.append(lam1)
    errors_list.append(1)

    while (errors_list[-1]) > tol and (acuatl_it < n_it_max):
        uk = np.dot(A, uk)
        lam_approx = np.dot(uk.T, np.dot(A, uk)) / (np.dot(uk.T, uk))

        err = abs(lam_approx - lam1)
        lam1 = lam_approx

        lambda_approx_list.append(lam_approx)
        errors_list.append(err)

        acuatl_it = acuatl_it + 1

    return uk, acuatl_it, errors_list, lambda_approx_list


def power_method_normalized(u0, A, tol, n_it_max):
    '''
        Normalized implementation of the power method used to find 
        an approximation of the eigen values. This method uses norms
        to make the algorithm numerically stable.

        ### INPUT
            - u0:start vector
            - A: matrix whose eigenvector will be calculated
            - t: tolerance 
            - n_it_max: this is the max number of iterations

        ### OUTPUT
            - zk: normalized eigenvector
            - actual_it: actual number of iteration before the stop
            - errors_list: a list that cotains the error of two consecutive eigen value lambda approximation
            - lamba_approx_list: list of approximantions of the eigen values
    '''
    acuatl_it = 0  # actual number of iterations

    z0 = u0 / np.linalg.norm(u0)  # u0 normalization
    # calculation of zk using u0 normalized: z0
    zk = np.dot(A, z0)/np.linalg.norm(np.dot(A, z0))
    lam1 = np.dot(zk.T, np.dot(A, zk)) / (np.dot(zk.T, zk))

    lambda_approx_list = []
    errors_list = []

    lambda_approx_list.append(lam1)
    errors_list.append(1)

    while (errors_list[-1]) > tol and (acuatl_it < n_it_max):
        # calculation of zk using u0 normalized: z0
        zk = np.dot(A, zk)/np.linalg.norm(np.dot(A, zk))
        lam_approx = np.dot(zk.T, np.dot(A, zk)) / (np.dot(zk.T, zk))

        err = abs(lam_approx - lam1)
        lam1 = lam_approx

        lambda_approx_list.append(lam_approx)
        errors_list.append(err)

        acuatl_it = acuatl_it + 1

    return zk, acuatl_it, errors_list, lambda_approx_list
