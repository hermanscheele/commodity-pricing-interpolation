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




def draw_random_interval_vector(I):
    n = len(I)
    v = []

    for i in range(n):
        high = I[i][1]
        low = I[i][2]
        draw = random.uniform(low, high)        
        v.append(draw)

    return v






from nelson_siegel_utils import fit_nelson_siegel, nelson_siegel
from data import y_avg, y_high, y_low, t
y_avg = np.array(y_avg)






I = construct_intervals(t, y_high, y_low)

def draw_gaussian_interval_vector(I):

    t_vals = [i[0] for i in I]
    t_vals = np.array(t_vals)

    print(y_avg)

    b0, b1, b2, g = fit_nelson_siegel(t_vals, y_avg)
    mean_vector = nelson_siegel(t_vals, b0, b1, b2, g)



    return mean_vector






draw_gaussian_interval_vector(I)


