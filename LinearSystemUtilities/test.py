import lu_mod as lu
import numpy as np

A1 = np.array([[-3., 2., 1.],
               [1., 0., -1.],
               [4., -2., 2.]])


print("\n-Partial pivoting")

(P1, L1, U1) = lu.LU_factorization_partial_pivoting(A1)
print("\nP1= \n", P1)
print("\nL1= \n", L1)
print("\nU1= \n", U1)

print("\nPAQ = LU")
print("\nPAQ:\n",P1*A1)
print("\nLU:\n",L1*U1)

print("\n-Complete pivoting")

(P1, L1, U1, Q1) = lu.LU_factorization_complete_pivoting(A1)
print("\nP1= \n", P1)
print("\nL1= \n", L1)
print("\nU1= \n", U1)

print("\nPAQ = LU")
print("\nPAQ:\n",P1*A1*Q1)
print("\nLU:\n",L1*U1)
