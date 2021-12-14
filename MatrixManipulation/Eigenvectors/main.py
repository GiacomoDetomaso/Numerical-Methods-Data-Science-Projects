import power_methods as pm
import numpy as np

print("Test of power method to find the eigen values")

A = np.array([[1, 2, 3], [1, 2, 7], [1, 3, 5]])  # input matrix

[uk, it, err_list, lam_list] = pm.power_method_std(
    np.array([1, 1, 1]), A, 1e-4, 100)

print("\n\nNumber of iterations: ", it)
print("Eigenvector: ", uk)
print("Final eigenvalue: ", lam_list[-1])
print("Final absolute error: ", err_list[-1])

print("\nNormalized method")

[uk, it, err_list, lam_list] = pm.power_method_normalized(
    np.array([1, 1, 1]), A, 1e-4, 100)

print("\nNumber of iterations: ", it)
print("Eigenvector: ", uk)
print("Final eigenvalue: ", lam_list[-1])
print("Final absolute error: ", err_list[-1])
