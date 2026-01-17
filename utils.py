import random
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


# derivative of f
def deriv(f, x, h):
    return (f(x + h) - f(x)) / h


# second derivative of f
def sec_deriv(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


def plot_f_integration(self, f, linspace, n):
    res = []
    for i in linspace:
        self.smooth = i
        I, _ = quad(f, 1, n)
        res.append(I)

    plt.plot(linspace, res, )


# Construct intervals for constraind NS regression
def construct_intervals(t_vals, high, low):

    n = len(t_vals)
    intervals = []

    for i in range(n):
        intervals.append( (t_vals[i], high[i], low[i]) )

    return intervals

