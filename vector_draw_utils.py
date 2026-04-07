import random
import numpy as np
from utils import construct_intervals
from nelson_siegel_utils import fit_nelson_siegel, constrained_fit_nelson_siegel, nelson_siegel
from data import y_mid, y_high, y_low, t



def R(x_k, x_l, rho):
    return np.exp(-0.5 * ((x_k - x_l) / rho)**2)


def construct_covariance(x, R, sigma_sq, rho):
    n = len(x)
    Sigma = np.zeros((n, n))
    for k in range(n):
        for l in range(n):
            Sigma[k, l] = sigma_sq * R(x[k], x[l], rho)
    return Sigma


def construct_cross_covariance(x_i, x_e, R, sigma_sq, rho):
    """Covariance between inequality locations x_i and equality locations x_e."""
    Sigma_ie = np.zeros((len(x_i), len(x_e)))
    for k in range(len(x_i)):
        for l in range(len(x_e)):
            Sigma_ie[k, l] = sigma_sq * R(x_i[k], x_e[l], rho)
    return Sigma_ie


def conditional_gaussian_params(x_i, x_e, z_e_hat, ns_params, sigma_sq, rho):
    """
    Compute conditional Gaussian parameters mu and Sigma for ϕ(z | ẑ^e).

    x_i       : illiquid maturities (n,)
    x_e       : liquid maturities (m,)
    z_e_hat   : observed liquid prices ẑ^e (m,)
    ns_params : (b0, b1, b2, g) for Nelson-Siegel trend Λ
    sigma_sq  : variance σ²
    rho       : correlation length ρ
    """
    # Trend means μ^i and μ^e  (eq. 6, 7)
    mu_i = nelson_siegel(x_i, *ns_params)
    mu_e = nelson_siegel(x_e, *ns_params)

    # Covariance matrices  (eq. 6, 7, 8)
    Sigma_i  = construct_covariance(x_i, R, sigma_sq, rho)          # (n x n)
    Sigma_e  = construct_covariance(x_e, R, sigma_sq, rho)          # (m x m)
    Sigma_ie = construct_cross_covariance(x_i, x_e, R, sigma_sq, rho)  # (n x m)

    Sigma_e_inv = np.linalg.inv(Sigma_e)

    # Conditional mean and covariance  (eq. 9, 10)
    mu    = mu_i + Sigma_ie @ Sigma_e_inv @ (z_e_hat - mu_e)
    Sigma = Sigma_i - Sigma_ie @ Sigma_e_inv @ Sigma_ie.T

    return mu, Sigma



def rejection_sample_gaussian(mu, Sigma, intervals, S):
    """
    Draw S samples from N(mu, Sigma) that satisfy all interval constraints.
    intervals: list of (t_i, low_i, high_i)
    """
    samples = []
    lows  = np.array([iv[1] for iv in intervals])
    highs = np.array([iv[2] for iv in intervals])

    while len(samples) < S:
        z = np.random.multivariate_normal(mu, Sigma)
        if np.all(z >= lows) and np.all(z <= highs):
            samples.append(z)
        
    return np.array(samples)  # shape (S, n)



