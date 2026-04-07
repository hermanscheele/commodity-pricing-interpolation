import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution


# Original Nelson-Siegel
def nelson_siegel(x, b0, b1, b2, g):
    return b0 + b1*np.exp(-g*x) + b2*x*np.exp(-g*x)


# Nelson-Siegel-Svensson extension 
def nelson_siegel_svensson(x, b0, b1, b2, b3, g1, g2):
    return (
            b0
            + b1 * np.exp(-g1 * x)
            + b2 * x * np.exp(-g1 * x)
            + b3 * x * np.exp(-g2 * x)
        )



# Original Nelson-Siegel fit
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



# Nelson-Siegel-Svensson fit
def fit_nelson_siegel_svensson(t, y):

    # ---- Initial guesses ----
    if np.any(t > 0):
        g1_init = 1.0 / np.median(t[t > 0])
    else:
        g1_init = 1.0
    g2_init = g1_init * 0.5    # second decay at a different rate

    b0_init = y[-1]
    b1_init = y[0] - y[-1]
    b2_init = 0.0
    b3_init = 0.0

    # Vector for intial curve-parameters
    p0 = [b0_init, b1_init, b2_init, b3_init, g1_init, g2_init]

    # ---- Bounds (g1, g2 > 0) ----
    bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, 1e-8, 1e-8],
              [ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf])

    # ---- Fit ----
    popt, _ = curve_fit(nelson_siegel_svensson, t, y, p0=p0, bounds=bounds, maxfev=10000)

    return popt   #  Returns optimal curve parameters [b0, b1, b2, b3, g1, g2] (NumPy array)




def constrained_fit_nelson_siegel(t, y, intervals, penalty_weight):

    def loss_function(theta):
        predictions = nelson_siegel(t, *theta)
        fit_error = np.sum((y - predictions)**2)
        penalty = 0
        for x_i, low_i, high_i in intervals:
            pred_i = nelson_siegel(x_i, *theta)
            penalty += max(0, low_i - pred_i)**2
            penalty += max(0, pred_i - high_i)**2
        return fit_error + penalty_weight * penalty

    bounds = [(-200, 200), (-200, 200), (-200, 200), (1e-6, 2.0)]
    res = differential_evolution(loss_function, bounds, maxiter=2000, tol=1e-10)
    return res.x



def constrained_fit_nelson_siegel_svensson(t, y, intervals, penalty_weight):

    def loss_function(theta):
        predictions = nelson_siegel_svensson(t, *theta)
        fit_error = np.sum((y - predictions)**2)
        penalty = 0
        for x_i, low_i, high_i in intervals:
            pred_i = nelson_siegel_svensson(x_i, *theta)
            penalty += max(0, low_i - pred_i)**2
            penalty += max(0, pred_i - high_i)**2
        return fit_error + penalty_weight * penalty

    bounds = [(-200, 200), (-200, 200), (-200, 200), (-200, 200), (1e-6, 2.0), (1e-6, 2.0)]
    res = differential_evolution(loss_function, bounds, maxiter=2000, tol=1e-10)
    return res.x