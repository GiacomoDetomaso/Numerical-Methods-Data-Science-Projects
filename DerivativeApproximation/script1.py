import numpy as np
import matplotlib.pyplot as plt

'''
Absolute error's study, in the first derivative's calculation
using the forward and center differences, considering f(x) = exp(x) and x0 = 1
'''

h = np.logspace(-16, 0, 1000) # set of points
x_0 = 1

df_1 = np.divide((np.exp(x_0 + h) - np.exp(x_0)), h) # forward differences' formula
df_2 = np.divide((np.exp(x_0 + h) - np.exp(x_0 - h)), 2 * h) # center differences' formula

# absolute error calculation
abs_err_df1 = abs(np.exp(x_0) - df_1)
abs_err_df2 = abs(np.exp(x_0) - df_2)

# error's graph with plot
plt.loglog(h, abs_err_df1, '.b', h, abs_err_df2, '.m')
plt.legend(['forward diff', 'center diff'])
plt.xlabel('h')
plt.ylabel('Absolute error')
plt.show()