import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from data import t, y, n

def normalize_Ti(t):
    norm = (t - min(t)) / (max(t) - min(t))
    return norm 


def nelson_siegel(x, b0, b1, b2, g):
    return b0 + b1*np.exp(-g*x) + b2*x*np.exp(-g*x)


def fit_nelson_siegel(t, y):
    
     # ---- Initial guesses ----
    if np.any(t > 0):
        gamma_init = 1.0 / np.median(t[t > 0])
    else:
        gamma_init = 1.0

    b0_init = y[-1]            # long end ~ last observed price
    b1_init = y[0] - y[-1]     # slope ~ difference short vs long
    b2_init = 0.0              # curvature starts at 0
    p0 = [b0_init, b1_init, b2_init, gamma_init]

    # ---- Bounds ----
    bounds = ([-np.inf, -np.inf, -np.inf, 1e-8],
              [ np.inf,  np.inf,  np.inf,  np.inf])

    # ---- Fit ----
    popt, _ = curve_fit(nelson_siegel, t, y, p0=p0, bounds=bounds, maxfev=10000)

    return popt   # (b0, b1, b2, g)


b0, b1, b2, g = fit_nelson_siegel(t, y)
t_fit = np.linspace(1, n, 100)
y_fit = nelson_siegel(t_fit, b0, b1, b2, g)

# ------- Plot Nelson Siegel fit --------
# plt.plot(t, y, 'o', color='r')
# plt.plot(t_fit, y_fit)
# plt.show()


# the correlation model p
def corr_func(h, smooth, h_squared: bool):
    if h_squared == 0: return np.exp(- smooth * h)
    else: return np.exp(- smooth * h**2)


# y_hat = F0(T_i) - m(T_i)
def construct_y_hat(t, y, b0, b1, b2, g):
    t = np.asarray(t, float).ravel()
    y = np.asarray(y, float).ravel()
    if t.shape != y.shape:
        raise ValueError("t and y must be same len")
        
    m = nelson_siegel(t, b0, b1, b2, g)
    return y - m

#print(construct_y_hat(t, y,b0, b1, b2, g))

