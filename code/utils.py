import numpy as np
from scipy.integrate import quad


# Derivative of f
def deriv(f, x, h):
    return (f(x + h) - f(x)) / h


# Second derivative of f
def sec_deriv(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


def integrate_over_linspace(f_of_smooth, linspace, a, b):
    res = []
    for s in linspace:
        I, _ = quad(f_of_smooth(s), a, b, limit=200)
        res.append(I)
    return np.array(res)


# Construct intervals for constraind NS regression
def construct_intervals(t_vals, high, low):
    n = len(t_vals)
    intervals = []
    for i in range(n):
        intervals.append( (t_vals[i], high[i], low[i]) )
    return intervals

