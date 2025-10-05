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



# the correlation model p
def corr_func(h, smooth, h_squared: bool):
    if h_squared == 0: return np.exp(- smooth * h)
    else: return np.exp(- smooth * h**2)


# y_hat = F0(T_i) - m(T_i)
def y_hat(t, y, b0, b1, b2, g):
    t = np.asarray(t, float).ravel()
    y = np.asarray(y, float).ravel()
    if t.shape != y.shape:
        raise ValueError("t and y must be same len")
        
    m = nelson_siegel(t, b0, b1, b2, g)
    return y - m

#print(y_hat(t, y,b0, b1, b2, g))


# construct c_x vector 
def c_x(x, n, t, smooth):
    c = []
    t = np.asarray(t, float).ravel()

    for k in range(n):
        abs_diff = abs(x - t[k])
        corr = corr_func(abs_diff, smooth, True) # correlation model fixed with False
        c.append(corr)
    
    return np.array(c)


# construct covariance matrix
def covar_matrix(n, t, smooth):
    matrix = []
    t = np.asarray(t, float).ravel()

    for i in range(n):
        row = []
        for j in range(n):
            abs_diff = abs(t[i] - t[j])
            corr = corr_func(abs_diff, smooth, True)
            row.append(corr)

        matrix.append(row)

    return np.array(matrix)
    

# Kriging fucntion
def kriging_func(x, t, b0, b1, b2, g, n, smooth):

    m = nelson_siegel(x, b0, b1, b2, g)
    cx = c_x(x, n, t, smooth)
    covar_mat_inv = np.linalg.inv(covar_matrix(n, t, smooth))
    y_h = y_hat(t, y, b0, b1, b2, g)

    return m + (cx.T @ covar_mat_inv @ y_h)





b0, b1, b2, g = fit_nelson_siegel(t, y)
t_fit = np.linspace(1, n, 400)
y_fit_regression = nelson_siegel(t_fit, b0, b1, b2, g)

# ------- Plot Nelson Siegel fit --------
plt.plot(t, y, 'o', color='r')
#plt.plot(t_fit, y_fit_regression)
# plt.show()

y_fit_gaussian = kriging_func(t_fit, t, b0, b1, b2, g, n, 1.0)
plt.plot(t_fit, y_fit_gaussian)
plt.show()
