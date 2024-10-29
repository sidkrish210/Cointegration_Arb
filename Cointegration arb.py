# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:09:48 2024

@author: Siddharth Krishnan
"""

import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import quantstats as qs
from arch.unitroot import ADF
from statsmodels.api import OLS
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
ticker = "AUDUSD=X"
ticker1 = "NZDUSD=X"

data_AUD = yf.download(ticker, start="2014-01-01", end="2023-01-01", interval="1d")
data_NZD = yf.download(ticker1, start="2014-01-01", end="2023-01-01", interval="1d")
data_AUD.index.name = 'Date'
data_NZD.index.name = 'Date'
data_AUD = pd.DataFrame(data_AUD)['Close']
data_NZD = pd.DataFrame(data_NZD)['Close']
df = pd.concat([data_AUD, data_NZD], axis=1)
df.columns = ['AUD', 'NZD']
df.plot(figsize=(10,7))
plt.ylabel("Price")
plt.show()
AUDUSD_adf = ADF(df['AUD'], trend="n", method="bic")
print("AUDUSD Augmented Dickey-Fuller Unit Root Test\n", AUDUSD_adf.regression.summary())
print("\nTest statistics and critical values: \n", AUDUSD_adf)

NZDUSD_adf = ADF(df['NZD'], trend="n", method="bic")
print("AUDUSD Augmented Dickey-Fuller Unit Root Test\n", NZDUSD_adf.regression.summary())
print("\nTest statistics and critical values: \n", NZDUSD_adf)

model = OLS(df.AUD.iloc[:90], df.NZD.iloc[:90])
model = model.fit() 
df['spread'] = df.AUD - model.params[0] * df.NZD
df.spread.plot(figsize=(10,7), color='g')
plt.ylabel("Spread")

plt.show()
spread_adf = ADF(df['spread'], trend="n", method="bic")
print("AUDUSD Augmented Dickey-Fuller Unit Root Test\n", spread_adf.regression.summary())
print("\nTest statistics and critical values: \n", spread_adf)

def mean_reversion(df, lookback, sigma):
    df['MA'] = df.spread.rolling(lookback).mean()
    df['MA_sd'] = df.spread.rolling(lookback).std()
    df['upper_band'] = df['MA'] + sigma*df['MA_sd']
    df['lower_band'] = df['MA'] - sigma*df['MA_sd']
    
    df['long_entry'] = df.spread<df['lower_band']
    df['Long_exit'] = df.spread>=df.MA
    df['position_long'] = np.nan
    df.loc[df.long_entry, 'position_long'] = 1
    df.loc[df.Long_exit, 'position_long'] = 0
    df.position_long = df.position_long.fillna(method='ffill')
    df['Short_entry'] = df.spread > df.upper_band
    df['Short_exit'] = df.spread <= df.MA
    df['position_short'] = np.nan
    df.loc[df.Short_entry, 'position_short'] = -1
    df.loc[df.Short_exit, 'position_short'] = 0
    df.position_short = df.position_short.fillna(method='ffill')
    df['positions'] = df.position_long + df.position_short

a = mean_reversion(df, 30, 2)
df['percentage_change'] = (df['spread']-df['spread'].shift(1))/(model.params[0]*df.NZD + df.AUD)
df['strategy_ret'] = df.positions.shift(1) * df.percentage_change
df['cum_returns'] = (df.strategy_ret + 1).cumprod()
print("The total strategy returns are %.2f" % ((df['cum_returns'].iloc[-1]-1)*100))
qs.reports.basic(df['strategy_ret']) 
df.cum_returns.plot(label = 'Returns', figsize = (10,7))
plt.xlabel('Date')
plt.ylabel('Cum_returs')
plt.show()