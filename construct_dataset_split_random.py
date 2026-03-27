import numpy as np
import matplotlib.pyplot as plt
from data import y_high, y_avg, y_low, n


t = np.linspace(1, n, len(y_avg))
half = len(t) // 2

# Randomly assign indices to point vs bid-ask
idx = np.random.permutation(len(t))
point_idx = np.sort(idx[:half])
bidask_idx = np.sort(idx[half:])

point_maturities = t[point_idx]
bidask_maturities = t[bidask_idx]

point_prices = y_avg[point_idx]
bid_prices = y_low[bidask_idx]
ask_prices = y_high[bidask_idx]


# Point prices
fig, ax = plt.subplots()
ax.scatter(point_maturities, point_prices)


# Bid-ask spreads with caps
cap = 0.1
for t, bid, ask in zip(bidask_maturities, bid_prices, ask_prices):
    ax.plot([t, t], [bid, ask], color="black")
    ax.plot([t - cap, t + cap], [bid, bid], color="black")
    ax.plot([t - cap, t + cap], [ask, ask], color="black")


plt.grid(alpha=0.4)
plt.show()
