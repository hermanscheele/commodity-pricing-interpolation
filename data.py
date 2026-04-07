import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = 15
data_shift = 0
df = pd.read_csv('data/wti_prices.csv')


y_high = df['high'][data_shift : n + data_shift]
y_avg = df['average'][data_shift : n + data_shift] # n open prices
y_low = df['low'][data_shift : n + data_shift]

y_mid = (y_high + y_low) / 2 

t = np.linspace(1, n, len(y_avg))


# plt.plot(t, y_high, 'o', label='high points')
# plt.plot(t, y_mid, 'o', color="green", label='mid points')
# plt.plot(t, y_low, 'o', color="red", label='low points')

# plt.vlines(t, y_low, y_high, color='gray', alpha=0.5, label='Daily Range (I_n)')

# plt.xlabel('t')
# plt.ylabel('price')

# plt.legend()
# plt.show()

