import numpy as np
import matplotlib.pyplot as plt
from nelson_siegel_utils import constrained_fit_nelson_siegel
from vector_draw_utils import conditional_gaussian_params, rejection_sample_gaussian
from kriging_utils import kriging_func
from data import t, y_mid, y_low, y_high


def split_liquid_illiquid(t, y_mid, y_low, y_high, threshold=0.0):
    t      = np.array(t)
    y_mid  = np.array(y_mid)
    y_low  = np.array(y_low)
    y_high = np.array(y_high)

    spread = y_high - y_low
    liquid_mask = spread <= threshold

    x_e       = t[liquid_mask]
    z_e_hat   = y_mid[liquid_mask]
    x_i       = t[~liquid_mask]
    intervals = [(t[k], y_low[k], y_high[k]) for k in range(len(t)) if not liquid_mask[k]]

    return x_e, z_e_hat, x_i, intervals


def sample_illiquid_prices(t, y_mid, y_low, y_high, threshold, sigma_sq, rho, S, penalty_w=1e6):
    # Split into liquid (equality) and illiquid (interval) points
    x_e, z_e_hat, x_i, intervals = split_liquid_illiquid(t, y_mid, y_low, y_high, threshold)
    print("done: 1")
    y_i_mid = np.array([(low + high) / 2 for _, low, high in intervals])

    # Fit constrained NS to all data with interval constraints at illiquid locations
    all_t     = np.concatenate([x_e, x_i])
    all_mid   = np.concatenate([z_e_hat, y_i_mid])
    sort_idx  = np.argsort(all_t)

    ns_params = constrained_fit_nelson_siegel(all_t[sort_idx], all_mid[sort_idx], intervals, penalty_w)
    print("done: 2")

    # Compute conditional Gaussian parameters ϕ(z | ẑ^e)
    mu, Sigma = conditional_gaussian_params(x_i, x_e, z_e_hat, ns_params, sigma_sq, rho)
    print("done: 3")

    # Draw S accepted samples via rejection sampling
    samples = rejection_sample_gaussian(mu, Sigma, intervals, S)
    print("done: 4")

    return samples, x_i, x_e, z_e_hat, ns_params, intervals



# ---------- Execute ---------- #
rho = 1.0
samples, x_i, x_e, z_e_hat, ns_params, intervals = sample_illiquid_prices(
    t, y_mid, y_low, y_high,
    threshold=1.0,
    sigma_sq=1.0,
    rho=rho,
    S=10,
)




smooth   = 1 / (2 * rho**2)
t_fit    = np.linspace(min(t), max(t), 500)
b0, b1, b2, g = ns_params

kriging_curves = []
for s in range(len(samples)):
    all_t = np.concatenate([x_e, x_i])
    all_y = np.concatenate([z_e_hat, samples[s]])
    sort_idx = np.argsort(all_t)
    t_obs = all_t[sort_idx]
    y_obs = all_y[sort_idx]
    curve = kriging_func(t_fit, t_obs, b0, b1, b2, g, len(t_obs), smooth, True, y_obs)
    kriging_curves.append(curve)

kriging_curves = np.array(kriging_curves)
mean_curve     = np.mean(kriging_curves, axis=0)




# ---------- Plot ---------- #
cap = 0.1
for curve in kriging_curves:
    plt.plot(t_fit, curve, color='blue', alpha=0.3)

plt.plot(t_fit, mean_curve, color='blue', linewidth=2, label='Mean kriging')
plt.scatter(x_e, z_e_hat, color='green', zorder=5, label='Liquid')

for j, (t_j, low_j, high_j) in enumerate(intervals):
    label = 'Illiquid' if j == 0 else None
    alpha = 0.5
    plt.plot([t_j, t_j], [low_j, high_j], color='red', alpha=alpha, label=label)
    plt.plot([t_j - cap, t_j + cap], [low_j, low_j], alpha=alpha, color='red')
    plt.plot([t_j - cap, t_j + cap], [high_j, high_j], alpha=alpha, color='red')

plt.legend()
plt.grid(alpha=0.3)
plt.xlabel(r'$t$')
plt.ylabel('Price')
plt.show()