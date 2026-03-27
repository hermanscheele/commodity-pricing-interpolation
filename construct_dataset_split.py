import numpy as np
import matplotlib.pyplot as plt
from data import y_high, y_avg, y_low, n


t = np.linspace(1, n, len(y_avg))
half = len(t) // 2

# Construct x-axis split of new (point) and late (bid-ask)
point_maturities = np.linspace(1, n, len(y_avg))[:(len(t) // 2)]
bidask_maturities = np.linspace(1, n, len(y_avg))[half:]


point_prices = y_avg[:half]
bid_prices = y_low[half:]
ask_prices = y_high[half:]

fig, ax = plt.subplots()

ax.scatter(point_maturities, point_prices)

for t, bid, ask in zip(bidask_maturities, bid_prices, ask_prices):
    ax.plot([t, t], [bid, ask], color="red")


plt.show()

