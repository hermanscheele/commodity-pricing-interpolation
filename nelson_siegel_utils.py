import numpy as np
from scipy.optimize import curve_fit, minimize


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





def constrained_fit_nelson_siegel(t, y, intervals):
    """
    Formal Constrained Regression:
    t: vector x^e (observed time points)
    y: vector z^e (observed prices/yields)
    intervals: I_1 x ... x I_n (the 'gates' for the trend)
    """
    
    # 1. THE LOSS FUNCTION: L(z^e, Lambda(x^e; theta))
    # This represents the sum of squared residuals between observations and the trend
    def loss_function(theta):
        # theta = [b0, b1, b2, g]
        # Lambda(x^e; theta)
        predictions = nelson_siegel(t, *theta)
        return np.sum((y - predictions)**2)

    # 2. THE CONSTRAINTS: Lambda(x_i; theta) \in I_i
    # Formally, for each i, low_i <= Lambda(x_i; theta) <= high_i
    cons = []
    for x_i, low_i, high_i in intervals:
        # Lower bound constraint: Lambda - low >= 0
        cons.append({'type': 'ineq', 'fun': lambda theta, x=x_i, l=low_i: nelson_siegel(x, *theta) - l})
        # Upper bound constraint: high - Lambda >= 0
        cons.append({'type': 'ineq', 'fun': lambda theta, x=x_i, h=high_i: h - nelson_siegel(x, *theta)})

    # 3. INITIAL GUESS (Warm Start)
    # To avoid the 'linesearch' error, start exactly where the unconstrained fit is
    p0 = fit_nelson_siegel(t, y)

    # 4. SOLVING THE PROBLEM
    # SLSQP is designed specifically for these inequality constraints
    res = minimize(
        loss_function, 
        p0, 
        method='SLSQP', 
        constraints=cons,
        bounds=[(None, None), (None, None), (None, None), (1e-6, 2.0)], # Ensure g > 0
        options={'ftol': 1e-10, 'maxiter': 2000}
    )

    return res.x