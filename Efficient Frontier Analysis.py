"""
Steps to build an efficient frontier

1. Data Preparation:
   - Select two stock tickers (AMD and NVDA).
   - Download 10 years of monthly historical closing prices using yfinance.
   - Calculate monthly returns, mean returns, and standard deviations.
   - Build a covariance matrix for portfolio variance calculations.

2. Portfolio Construction:
   - Set a range from 0 to 1 (AMD vs NVDA) for portfolio weights.
   - Calculate expected portfolio return and standard deviation for each weight.
   - Store results in lists for plotting.

3. Visualization (Efficient Frontier):
   - Set the x axis as risk and y axis as return
   - Scatter plot of all portfolio combinations (risk vs return).
   - Connect the points to show the efficient frontier curve.

4. Customize the appearance.
    - Include different parameters so that the graph is readable.

5. Additional Plot Feature
   - Highlight the Minimum Variance Portfolio (MVP) with a red marker.
   - Add annotation to clearly label the MVP on the plot.

"""

# ==== Task 1: Creating Efficient Frontier Plot ====

# Importing required libraries

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Preparation

TICKERS = ['AMD', 'NVDA']   # choosing two stock tickers
SAMPLE_LEN = '10y'          # 10 years of historical data

data = []               # will hold per-stock return & stdev
df = pd.DataFrame()     # will store monthly returns of both stocks

for tic in TICKERS:
    price = (
        yf.Ticker(tic)
        .history(SAMPLE_LEN)['Close']   # daily close prices
        .resample('1ME')                # resample to monthly (month end)
        .last()                         # take last closing price each month
    )
    monthly_ret = price.pct_change()    # calculate monthly returns
    df[tic] = monthly_ret.dropna()      # Adding monthly return to a DataFrame
    rets = monthly_ret.mean()           # average monthly return
    stdev = monthly_ret.std()           # standard deviation of monthly return
    
    # Saving per-stock stats (return & stdev)
    data.append({'Ticker': tic, 'ret': rets, 'stdev': stdev})

# Creating a summary table of stock statistics
df_data = pd.DataFrame(data).set_index('Ticker')

# Covariance matrix of monthly returns 
# Needed for portfolio standard deviation formula
cov_mtx = df.cov()

# Portfolio Construction with Weights

weights = np.arange(0,101) / 100    # AMD weights ranging from 0% to 100%

x_axis = []     # portfolio stdev values (risk)
y_axis = []     # portfolio return values

for w in weights:
# Setting the formulae for the portfolio return and standard deviation

    # Portfolio expected return = weighted average of stock returns
    # E(R) = w₁r₁ + w₂r₂
    port_ret = w * df_data.loc['AMD','ret'] + (1-w) * df_data.loc['NVDA','ret']

    # Portfolio standard deviation using variance–covariance formula:
    # σp= sqrt(w₁²σ1² + w₂²σ2² + 2w₁w₂Cov12)
    port_stdev = np.sqrt(
        w**2 * cov_mtx.loc['AMD','AMD'] +
        (1-w)**2 * cov_mtx.loc['NVDA','NVDA'] +
        2 * w * (1-w) * cov_mtx.loc['NVDA','AMD']
    )
    x_axis.append(port_stdev)
    y_axis.append(port_ret)

# Visualizing Efficient Frontier
fig, ax = plt.subplots()

# Scatter plot of portfolio combinations
ax.scatter(x_axis,y_axis, s = 5)

# ==== Task 2: Customizing the Plot ====

# Adding title and axis labels
ax.set_title('Efficient Frontier of AMD & NVDA', loc = 'center')
ax.set_xlabel('Standard Deviation')
ax.set_ylabel('Expected Return')

# Plotting the frontier line (connecting points)
ax.plot(x_axis, y_axis, color = 'black', linestyle = '-', linewidth = 2)

# Adding grid for readability
ax.grid(True)

# === Task 3: Additional Feature (Minimum Variance Portfolio) ===
# Finding the index of the lowest standard deviation point
i_min = np.argmin(x_axis)

# Specifying the minimum variance point in the graph
min_var_std = x_axis[i_min]
min_var_ret = y_axis[i_min]

# Highlighting Minimum Variance point with a red marker
plt.scatter([min_var_std], [min_var_ret], color='red', s=50)

# Annotating Minimum Variance Point so that it can be easily identified
ax.annotate('Mimimum Variance Portfolio', 
            xy = (min_var_std, min_var_ret),
            xytext = (5,0),
            textcoords = 'offset points')

# The final output
plt.show()
