import numpy as np
import matplotlib.pyplot as plt

# Bisection method considering  y = exp(x) - x

fun = (lambda x: np.exp(-x) - x)  # starting function
a = -5
b = 5

pts = np.linspace(a, b, 500)
print("Points of the function:\n {}".format(pts))

# graphs
# design the graph of the function
plt.plot(pts, fun(pts), "-k", label='$f(x)$')
plt.plot(pts, 0 * pts, 'b', label='$y=0$')  # design y = 0 graph
# a single point which is an approximate root of the function
plt.plot(0.5, 0, 'pr', label='roots of $f(x)$')
plt.legend()
plt.show()


def bisection(f, a, b, tol):
    '''
    ## Bisection method to find function's roots. 
    ### Input:
        - f: the function where to find the roots; 
        - a: point a; 
        - b: point b;
        - tol: a tolerence;

    ### Output:
        - c: the root after all the iteration;
        - mid_succesion: a list which contains all the midpoints; 
        - it: the number of iterations;
        - flag: true if 'it' is <= of the theorical number of iterations, false otherwise;
    '''

    f_a = f(a)  # image of f in a
    f_b = f(b)  # image of f in b
    midp_succession = []  # this list will contain all the values of c

    if f_a * f_b < 0:
        # theoric number of iterations
        min_it = np.ceil(np.log2((b-a)/(2*tol)))
        it = 1  # actual number of iterations

        c = (a + b)/2  # midpoint
        f_c = f(c)  # image of f in c
        midp_succession.append(c)

        # bisection loop
        while abs(f_c) > tol:
            it = it + 1
            if f_a * f(c) < 0:
                b = c
            else:
                a = c
                f_a = f(a)

            c = (a + b)/2
            f_c = f(c)
            midp_succession.append(c)

        if it >= min_it:
            flag = True
        else:
            flag = False

        return (c, midp_succession, it, flag)
    else:
        print("Error")


(root, succession, iterations_number, flag) = bisection(fun, a, b, tol=1e-3)

# print the result set
print("-Root of y = exp(-x) - x: {}".format(root))
print("-Value of f in root: {}".format(fun(root)))
print("-Number of iterations: {}".format(iterations_number))
print("-Does the number of iterations respect the theorical one? {}".format(flag))

# error calculation
array_np = np.array(succession)  # translate the list into a numpy array
size = len(array_np)

# the error is a difference of two successive approximations (c_n - c_n-1) where c_n is the list of midpoints
error_approximation = abs(array_np[1:size] - array_np[0:size - 1])
plt.semilogy(np.linspace(2, iterations_number,
             iterations_number-1), error_approximation, '-ob')
plt.show()
