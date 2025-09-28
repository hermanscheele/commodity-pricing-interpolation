import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('archive/wti_prices.csv')

o_p = df['open'][:15] # 30 open prices

x = np.linspace(1, 15, len(o_p))
y = np.array(o_p)

# plt.plot(x, y, 'o')
# plt.show()
