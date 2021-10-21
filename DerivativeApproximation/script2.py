import numpy as np
import matplotlib.pyplot as plt

'''
Relative error's study, in the first derivative's calculation
using the forward and center differences, considering:
    - f(x) = log(x) + x/2
    - h_i = 2^(-i) with 0 <= i <= 50 
    - x0 = 15
'''

base = 2
x_0 = 15

# lamba function to define f(x) = log(x) + x/2
fun = (lambda x: np.log(x) + np.divide(x, 2))
fun_prime = (lambda x: np.divide(1, x) + 1/2)  # first derivative of fun

i = np.linspace(1, 50, 50)  # all the values of i are stored in an array
h = np.power(2, -i) # element wise power

# approximations' calculations in x0
df_1 = np.divide((fun(x_0 + h) - fun(x_0)), h) # forward difference's formula
df_2 = np.divide(fun(x_0 + h) - fun(x_0 - h), 2 * h) # center difference's formula

# calculation of the relative errors commited during the approximarion
err_fun__a = abs(fun_prime(x_0) - df_1)/abs(fun_prime(x_0)) # relative error in forward difference
err_fun_c = abs(fun_prime(x_0) - df_2)/abs(fun_prime(x_0)) # relative error in center difference

# error's graph with plot
plt.figure
plt.loglog(h, err_fun__a, '.b', h, err_fun_c, '.m')
plt.legend(['forward diff', 'center diff'])
plt.xlabel('h')
plt.ylabel('Relative error')
plt.show()
