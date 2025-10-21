import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = 20
df = pd.read_csv('archive/wti_prices.csv')

o_p = df['open'][:n] # n open prices

t = np.linspace(1, n, len(o_p))
y = np.array(o_p)
