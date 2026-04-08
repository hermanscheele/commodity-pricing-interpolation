# commodity-pricing-interpolation
Install dependencies
```
pip install -r requirements.txt
```

### Data
- `brent_prices.py`
- `wti_prices.py`


Downloaded from: https://www.kaggle.com/datasets/nikitamanaenkov/historical-crude-oil-futures-prices-wti-and-brent

License: Apache 2.0

Change `data_shift` variable in `data.py` to explore different timestamp intervals.

### kriging.py
Kriging class with parameter updates, mutiple fitting paradimgs and plotting scripts. Execute:

```
python3 kriging.py
```

### sample_illiquid_prices.py
Generates a liquid (exact) and illiquid (bid-ask) data split -> Fetches the conditional gaussian distribution -> runs rejection sampling -> computes average curve -> plotting. Execute:

```
python3 sample_illiquid_prices.py
```

### kriging_constrained.py
Soon to be, general class file for Kriging with Data Augumentation. Under construction.