
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = 20
df = pd.read_csv('data/wti_prices.csv')

o_p = df['open'][:n] # n open prices

t = np.linspace(1, n, len(o_p))
y = np.array(o_p)


# plt.plot(t, y, 'o', label='price points')
# plt.xlabel('t')
# plt.ylabel('price')

# plt.legend()
# plt.show()