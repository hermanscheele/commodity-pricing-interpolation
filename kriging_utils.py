import numpy as np
from nelson_siegel_utils import nelson_siegel

# the correlation model p
def corr_func(h, smooth, h_squared: bool):
    if h_squared == False: return np.exp(- smooth * h)
    else: return np.exp(- smooth * h**2)


# y_hat = F0(T_i) - m(T_i)
def y_hat(t, y, b0, b1, b2, g):
    t = np.asarray(t, float).ravel()
    y = np.asarray(y, float).ravel()
    if t.shape != y.shape:
        raise ValueError("t and y must be same len")
        
    m = nelson_siegel(t, b0, b1, b2, g)
    return y - m


# construct c_x vector 
def c_x(x, n, t, smooth, gauss):
    c = []
    t = np.asarray(t, float).ravel()

    for k in range(n):
        abs_diff = abs(x - t[k])
        corr = corr_func(abs_diff, smooth, gauss) 
        c.append(corr)
    
    return np.array(c)


# construct covariance matrix
def covar_matrix(n, t, smooth, gauss):
    matrix = []
    t = np.asarray(t, float).ravel()

    for i in range(n):
        row = []
        for j in range(n):
            abs_diff = abs(t[i] - t[j])
            corr = corr_func(abs_diff, smooth, gauss)
            row.append(corr)

        matrix.append(row)

    return np.array(matrix)


# Kriging fucntion
def kriging_func(x, t, b0, b1, b2, g, n, smooth, gauss, y):

    m = nelson_siegel(x, b0, b1, b2, g)
    cx = c_x(x, n, t, smooth, gauss)
    covar_mat_inv = np.linalg.inv(covar_matrix(n, t, smooth, gauss))
    y_h = y_hat(t, y, b0, b1, b2, g)

    return m + (cx.T @ covar_mat_inv @ y_h)
