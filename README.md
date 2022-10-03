# Mean-reversion-trading-strategy
An implementation of mean reversion trading strategy using classic TA indicators modified with EOT filter

---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.6
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .markdown}
# Ranton Mean Reversion: Study 1

Use the \"Run\" button to execute the code.
:::

::: {.cell .markdown heading_collapsed="true"}
### Importing libraries & packages {#importing-libraries--packages}
:::

::: {.cell .code execution_count="1" hidden="true"}
``` {.python}
!pip install jovian ccxt backtesting pyti dateparser pandas_ta plotly --upgrade --quiet
```
:::

::: {.cell .code execution_count="2" hidden="true"}
``` {.python}
import ccxt
import jovian
import requests
import math
import datetime
import concurrent
import os
import glob
import time
import dateparser

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px
import plotly.graph_objects as go

import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import adfuller
import scipy.optimize as spop
from datetime import datetime, timedelta

from pyti import bollinger_bands
from pyti.average_true_range import average_true_range as average_true_range
from pyti import exponential_moving_average
```
:::

::: {.cell .markdown}
### Enter Symbols:
:::

::: {.cell .code execution_count="3"}
``` {.python}
s1= ['DASHUSDT', 'ICXUSDT', 'DASHUSDT',
     'DASHUSDT','VETUSDT', 'SOLUSDT', 'DASHUSDT',
     'SOLUSDT', 'OMGUSDT', 'DASHUSDT', 'DASHUSDT']

s2 = ['VETUSDT', 'IOTAUSDT', 'IOSTUSDT',
     'DOTUSDT', 'ICXUSDT', 'DOTUSDT', 'HBARUSDT',
     'VETUSDT', 'QTUMUSDT', 'SOLUSDT', 'LTCUSDT']

coins = ['DASHUSDT','SOLUSDT']
timeinterval = '4h'
```
:::

::: {.cell .code execution_count="4"}
``` {.python}
for i,j in zip(s1,s2):
        print(f'{i}PERP - {j}PERP')
```

::: {.output .stream .stdout}
    DASHUSDTPERP - VETUSDTPERP
    ICXUSDTPERP - IOTAUSDTPERP
    DASHUSDTPERP - IOSTUSDTPERP
    DASHUSDTPERP - DOTUSDTPERP
    VETUSDTPERP - ICXUSDTPERP
    SOLUSDTPERP - DOTUSDTPERP
    DASHUSDTPERP - HBARUSDTPERP
    SOLUSDTPERP - VETUSDTPERP
    OMGUSDTPERP - QTUMUSDTPERP
    DASHUSDTPERP - SOLUSDTPERP
    DASHUSDTPERP - LTCUSDTPERP
:::
:::

::: {.cell .markdown heading_collapsed="true"}
### Data collection & cleaning: {#data-collection--cleaning}
:::

::: {.cell .code execution_count="5" code_folding="[3]" hidden="true"}
``` {.python}
binance = ccxt.binanceusdm({'options': {'enableRateLimit': True}})

def getdata(symbol, interval):
    '''Function to fetch historical closing prices of futures ticker
        symbol: futures ticker symbol as string
        interval: timeframe as string '5m', '1h','1d'
    '''
    
    data = pd.DataFrame(binance.fetch_ohlcv( symbol, interval, limit=1500))
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    data['Date'] = pd.to_datetime(data['Date'], unit='ms') + pd.Timedelta(hours=2)
    data = data.set_index('Date')
    data = data.drop(['Open', 'High', 'Low', 'Volume'], axis=1)
    data.rename(columns={'Close': f'{symbol}'}, inplace=True)
    
    return data

dfs=[]
interval = timeinterval


for coin in coins:
    dfs.append(getdata(coin, interval))
    
data = pd.concat(dfs, axis=1)
returns = np.log(data.pct_change()+1)
returns = returns.iloc[1:,:]
data = data.iloc[1:,:]
#data = data.loc[year:]
data = data.dropna()

gross_returns = np.array([])
net_returns = np.array([])
t_s = np.array([])
coin1 = coins[0]
coin2 =  coins[1]

data['close'] = data[coin1]- data[coin2]
```
:::

::: {.cell .markdown heading_collapsed="true"}
### Bollinger Bands dataframe:
:::

::: {.cell .code execution_count="6" hidden="true"}
``` {.python}
df_bb  = data.copy()
df_bb['upper'] = bollinger_bands.upper_bollinger_band(df_bb.close, period=20, std_mult=1.9)
df_bb['lower'] = bollinger_bands.lower_bollinger_band(df_bb.close, period=20, std=1.9)
df_bb['above_upper_bb'] = df_bb.close > df_bb.upper
df_bb['below_lower_bb'] = df_bb.close < df_bb.lower
df_bb.tail()
```

::: {.output .execute_result execution_count="6"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DASHUSDT</th>
      <th>SOLUSDT</th>
      <th>close</th>
      <th>upper</th>
      <th>lower</th>
      <th>above_upper_bb</th>
      <th>below_lower_bb</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-08-23 02:00:00</th>
      <td>47.01</td>
      <td>35.20</td>
      <td>11.81</td>
      <td>11.994950</td>
      <td>8.618050</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 06:00:00</th>
      <td>46.90</td>
      <td>34.94</td>
      <td>11.96</td>
      <td>12.172069</td>
      <td>8.738931</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 10:00:00</th>
      <td>47.98</td>
      <td>35.66</td>
      <td>12.32</td>
      <td>12.348383</td>
      <td>8.918617</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 14:00:00</th>
      <td>47.92</td>
      <td>35.69</td>
      <td>12.23</td>
      <td>12.451200</td>
      <td>9.154800</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 18:00:00</th>
      <td>47.98</td>
      <td>35.30</td>
      <td>12.68</td>
      <td>12.658998</td>
      <td>9.284002</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown hidden="true"}

Price closes above the upper bollinger band:
:::

::: {.cell .code execution_count="7" hidden="true"}
``` {.python}
df_bb[(df_bb.above_upper_bb == True)].tail()
```

::: {.output .execute_result execution_count="7"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DASHUSDT</th>
      <th>SOLUSDT</th>
      <th>close</th>
      <th>upper</th>
      <th>lower</th>
      <th>above_upper_bb</th>
      <th>below_lower_bb</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-08-17 18:00:00</th>
      <td>53.15</td>
      <td>40.72</td>
      <td>12.43</td>
      <td>11.423100</td>
      <td>8.530900</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-17 22:00:00</th>
      <td>53.04</td>
      <td>40.51</td>
      <td>12.53</td>
      <td>11.862065</td>
      <td>8.294935</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-18 02:00:00</th>
      <td>53.46</td>
      <td>40.80</td>
      <td>12.66</td>
      <td>12.285692</td>
      <td>8.126308</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-22 14:00:00</th>
      <td>46.43</td>
      <td>34.78</td>
      <td>11.65</td>
      <td>11.381344</td>
      <td>8.549656</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 18:00:00</th>
      <td>47.98</td>
      <td>35.30</td>
      <td>12.68</td>
      <td>12.658998</td>
      <td>9.284002</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown hidden="true"}

Price closes below the lower bollinger band:
:::

::: {.cell .code execution_count="8" hidden="true"}
``` {.python}
df_bb[(df_bb.below_lower_bb == True)].tail()
```

::: {.output .execute_result execution_count="8"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DASHUSDT</th>
      <th>SOLUSDT</th>
      <th>close</th>
      <th>upper</th>
      <th>lower</th>
      <th>above_upper_bb</th>
      <th>below_lower_bb</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-07-30 18:00:00</th>
      <td>51.75</td>
      <td>45.07</td>
      <td>6.68</td>
      <td>11.150247</td>
      <td>6.896753</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2022-08-12 22:00:00</th>
      <td>56.09</td>
      <td>45.46</td>
      <td>10.63</td>
      <td>13.016932</td>
      <td>11.098068</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2022-08-13 02:00:00</th>
      <td>56.78</td>
      <td>47.40</td>
      <td>9.38</td>
      <td>13.375555</td>
      <td>10.453445</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2022-08-13 06:00:00</th>
      <td>56.44</td>
      <td>47.17</td>
      <td>9.27</td>
      <td>13.548182</td>
      <td>9.953818</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2022-08-13 10:00:00</th>
      <td>56.12</td>
      <td>46.73</td>
      <td>9.39</td>
      <td>13.654852</td>
      <td>9.579148</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown heading_collapsed="true"}
### ATR Bands dataframe:
:::

::: {.cell .code execution_count="9" hidden="true"}
``` {.python}
df_atr = data.copy()
atr_multiplier = 1.5
df_atr['atr'] = average_true_range(df_atr.close,15)
df_atr['sell_stop_loss'] = (df_atr.close + df_atr['atr'] * atr_multiplier) #atr_upper_band
df_atr['buy_stop_loss'] = (df_atr.close - df_atr['atr'] * atr_multiplier) #atr_lower_band
df_atr.tail()
```

::: {.output .execute_result execution_count="9"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DASHUSDT</th>
      <th>SOLUSDT</th>
      <th>close</th>
      <th>atr</th>
      <th>sell_stop_loss</th>
      <th>buy_stop_loss</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-08-23 02:00:00</th>
      <td>47.01</td>
      <td>35.20</td>
      <td>11.81</td>
      <td>2.597411</td>
      <td>15.706116</td>
      <td>7.913884</td>
    </tr>
    <tr>
      <th>2022-08-23 06:00:00</th>
      <td>46.90</td>
      <td>34.94</td>
      <td>11.96</td>
      <td>2.558917</td>
      <td>15.798375</td>
      <td>8.121625</td>
    </tr>
    <tr>
      <th>2022-08-23 10:00:00</th>
      <td>47.98</td>
      <td>35.66</td>
      <td>12.32</td>
      <td>2.546989</td>
      <td>16.140484</td>
      <td>8.499516</td>
    </tr>
    <tr>
      <th>2022-08-23 14:00:00</th>
      <td>47.92</td>
      <td>35.69</td>
      <td>12.23</td>
      <td>2.535856</td>
      <td>16.033785</td>
      <td>8.426215</td>
    </tr>
    <tr>
      <th>2022-08-23 18:00:00</th>
      <td>47.98</td>
      <td>35.30</td>
      <td>12.68</td>
      <td>2.549466</td>
      <td>16.504199</td>
      <td>8.855801</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown heading_collapsed="true"}
### 200 Period EMA dataframe: {#200-period-ema-dataframe}
:::

::: {.cell .code execution_count="10" hidden="true" scrolled="false"}
``` {.python}
df_ema = data.copy()
df_ema['ema_200'] = exponential_moving_average.exponential_moving_average(df_ema.close, period=200)
df_ema['price_above_ema_200'] = df_ema.ema_200 < df_ema.close
df_ema['price_below_ema_200'] = df_ema.ema_200 > df_ema.close

df_ema.tail()#.iloc[1324:1330]
```

::: {.output .execute_result execution_count="10"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DASHUSDT</th>
      <th>SOLUSDT</th>
      <th>close</th>
      <th>ema_200</th>
      <th>price_above_ema_200</th>
      <th>price_below_ema_200</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-08-23 02:00:00</th>
      <td>47.01</td>
      <td>35.20</td>
      <td>11.81</td>
      <td>10.384687</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 06:00:00</th>
      <td>46.90</td>
      <td>34.94</td>
      <td>11.96</td>
      <td>10.411092</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 10:00:00</th>
      <td>47.98</td>
      <td>35.66</td>
      <td>12.32</td>
      <td>10.440396</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 14:00:00</th>
      <td>47.92</td>
      <td>35.69</td>
      <td>12.23</td>
      <td>10.468186</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 18:00:00</th>
      <td>47.98</td>
      <td>35.30</td>
      <td>12.68</td>
      <td>10.501983</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown hidden="true"}

Price closes above the ema line:
:::

::: {.cell .code execution_count="11" hidden="true"}
``` {.python}
df_ema[(df_ema.price_above_ema_200 == True)].tail()
```

::: {.output .execute_result execution_count="11"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DASHUSDT</th>
      <th>SOLUSDT</th>
      <th>close</th>
      <th>ema_200</th>
      <th>price_above_ema_200</th>
      <th>price_below_ema_200</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-08-23 02:00:00</th>
      <td>47.01</td>
      <td>35.20</td>
      <td>11.81</td>
      <td>10.384687</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 06:00:00</th>
      <td>46.90</td>
      <td>34.94</td>
      <td>11.96</td>
      <td>10.411092</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 10:00:00</th>
      <td>47.98</td>
      <td>35.66</td>
      <td>12.32</td>
      <td>10.440396</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 14:00:00</th>
      <td>47.92</td>
      <td>35.69</td>
      <td>12.23</td>
      <td>10.468186</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 18:00:00</th>
      <td>47.98</td>
      <td>35.30</td>
      <td>12.68</td>
      <td>10.501983</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown hidden="true"}

Price closes below the ema line:
:::

::: {.cell .code execution_count="12" hidden="true"}
``` {.python}
df_ema[(df_ema.price_below_ema_200 == True)].tail()
```

::: {.output .execute_result execution_count="12"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DASHUSDT</th>
      <th>SOLUSDT</th>
      <th>close</th>
      <th>ema_200</th>
      <th>price_above_ema_200</th>
      <th>price_below_ema_200</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-08-20 10:00:00</th>
      <td>45.88</td>
      <td>36.57</td>
      <td>9.31</td>
      <td>10.183788</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2022-08-20 14:00:00</th>
      <td>46.12</td>
      <td>36.48</td>
      <td>9.64</td>
      <td>10.182333</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2022-08-21 14:00:00</th>
      <td>46.22</td>
      <td>35.99</td>
      <td>10.23</td>
      <td>10.241952</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2022-08-22 02:00:00</th>
      <td>46.14</td>
      <td>36.08</td>
      <td>10.06</td>
      <td>10.274293</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2022-08-22 06:00:00</th>
      <td>44.83</td>
      <td>34.89</td>
      <td>9.94</td>
      <td>10.278317</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown heading_collapsed="true"}
### EOT indicator:
:::

::: {.cell .code execution_count="13" code_folding="[3,18,34,53,64]" hidden="true"}
``` {.python}
#Boom-Pro Indicator

#Input Close values to return HP
def high_pass_filter(data):
    PI = np.pi
    HP = []

    for i, _ in enumerate(data):
        if i < 2:
            HP.append(0)
        else:
            angle = 0.707 * 2 * PI/100
            alpha1 = (math.cos(angle) + math.sin(angle)-1)/ math.cos(angle)
            HP.append(math.pow(1.0-alpha1/2.0, 2)*(data[i]-2*data[i-1]+data[i-2]) + 2*(1-alpha1)*HP[i-1] - math.pow(1-alpha1, 2)*HP[i-2])

    return HP

#Input HP to smoothen and return Filt 
def super_smoother(data, LPPeriod):
    Filt = []
    for i, _ in enumerate(data):
        if i < 2:
            Filt.append(0)
        else:
            arg = 1.414 * 3.14159 / LPPeriod
            a_1 = math.exp(-arg)
            b_1 = 2 * a_1 * math.cos(4.44/float(LPPeriod))
            c_2 = b_1
            c_3 = -a_1 * a_1
            c_1 = 1 - c_2 - c_3
            Filt.append(c_1 * (data[i] + data[i-1]) / 2 + c_2 * Filt[i-1] + c_3 * Filt[i-2])
    return Filt

#Input Filt values to return X
def agc(data):
    X = []
    Peak = []
    for i, _ in enumerate(data):
        if i < 1:
            X.append(0)
            Peak.append(0)
            #Peak.append(.0000001)
        else:
            Peak.append(0.991 * Peak[i - 1])
            if abs(data[i]) > Peak[i]:
                Peak[i] = abs(data[i])

            if Peak[i] != 0:
                X.append(data[i] / Peak[i])

    return X

#Input X to return final Quotient 
def quotient(data, K_val):
    K = K_val
    Q = []
    for i, _ in enumerate(data):
        if i<1:
            Q.append(0)
        else:
            Q.append((data[i]+ K) / (K*data[i]+1))
    return Q

#Input Quotient to return smoothen the quotient
def sma(data, length):
    trigger = []
    for i, _ in reversed(list(enumerate(data))):
        sum = 0
        for t in range(i - length + 1, i + 1):
            sum = sum + data[t] / length
        trigger.insert(0, sum)
    return trigger
```
:::

::: {.cell .code execution_count="14" hidden="true"}
``` {.python}
df_eot = data.copy()

K1 = 0
K2 = 0.4
LPPeriod_1 = 6
LPPeriod_2 = 27

HP = high_pass_filter(df_eot.close)

#quotient 1 params
Filt_1 = super_smoother(HP, LPPeriod_1)
X_1 = agc(Filt_1)

#quotient 2 params
Filt_2 = super_smoother(HP, LPPeriod_2)
X_2 = agc(Filt_2)

df_eot = df_eot.reset_index()
q1 = quotient(X_1, K1)
q2 = quotient(X_2, K2)

trig = sma(q1,length=2)
df_eot['trig'] = pd.Series(trig)

df_eot['white_line'] = df_eot['trig']
df_eot['red_line'] = pd.Series(q2)

df_eot['white_line'] = (df_eot['white_line']*100)+10
df_eot['red_line'] = (df_eot['red_line']*100)+10


df_eot['prev_white_line'] = df_eot['white_line']
df_eot['prev_white_line'] = df_eot['prev_white_line'].shift(1)

df_eot['prev_red_line'] = df_eot['red_line']
df_eot['prev_red_line'] = df_eot['prev_red_line'].shift(1)

#df_eot = df_eot.fillna(method='ffill')
#df_eot.dropna(inplace=True)

df_eot['bullish_cross'] = (df_eot['prev_white_line'] < df_eot['red_line']) & (df_eot['white_line'] > df_eot['red_line']) & (df_eot['white_line']<85)

df_eot['bearish_cross'] = (df_eot['prev_red_line'] < df_eot['white_line']) & (df_eot['red_line'] > df_eot['white_line']) & (df_eot['red_line']>15)
df_eot['crossover'] = df_eot.apply(lambda x: 'bullish crossover' if x.bullish_cross == True
                      else ('bearish crossover' if x.bearish_cross == True else 'none'), axis=1)
df_eot = df_eot.drop(['bullish_cross','bearish_cross'], axis=1)
df_eot = df_eot[df_eot.columns.drop(list(df_eot.filter(regex='index')))]

df_eot['top_level_eot'] = (df_eot.prev_white_line >55) & (df_eot.red_line>55)
df_eot['bottom_level_eot'] = (df_eot.prev_white_line <45) & (df_eot.red_line<45)

df_eot = df_eot.set_index('Date')
df_eot.tail()
```

::: {.output .execute_result execution_count="14"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DASHUSDT</th>
      <th>SOLUSDT</th>
      <th>close</th>
      <th>trig</th>
      <th>white_line</th>
      <th>red_line</th>
      <th>prev_white_line</th>
      <th>prev_red_line</th>
      <th>crossover</th>
      <th>top_level_eot</th>
      <th>bottom_level_eot</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-08-23 02:00:00</th>
      <td>47.01</td>
      <td>35.20</td>
      <td>11.81</td>
      <td>0.787907</td>
      <td>88.790738</td>
      <td>89.695698</td>
      <td>85.025340</td>
      <td>81.772969</td>
      <td>bearish crossover</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 06:00:00</th>
      <td>46.90</td>
      <td>34.94</td>
      <td>11.96</td>
      <td>0.809204</td>
      <td>90.920390</td>
      <td>97.024559</td>
      <td>88.790738</td>
      <td>89.695698</td>
      <td>bearish crossover</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 10:00:00</th>
      <td>47.98</td>
      <td>35.66</td>
      <td>12.32</td>
      <td>0.840041</td>
      <td>94.004106</td>
      <td>103.553014</td>
      <td>90.920390</td>
      <td>97.024559</td>
      <td>none</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 14:00:00</th>
      <td>47.92</td>
      <td>35.69</td>
      <td>12.23</td>
      <td>0.868231</td>
      <td>96.823142</td>
      <td>109.299294</td>
      <td>94.004106</td>
      <td>103.553014</td>
      <td>none</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-08-23 18:00:00</th>
      <td>47.98</td>
      <td>35.30</td>
      <td>12.68</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>96.823142</td>
      <td>109.299294</td>
      <td>none</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown heading_collapsed="true"}
### Six Candle Close dataframe:
:::

::: {.cell .code execution_count="15" hidden="true" scrolled="false"}
``` {.python}
df_6 = data.copy()
df_6['six_greater'] = (df_6['close'].rolling(window=7, min_periods=1)
             .apply(lambda w: w.iloc[:-1].lt(w.iloc[-1]).all()))

df_6['six_lower'] = (df_6['close'].rolling(window=7, min_periods=1)
             .apply(lambda w: w.iloc[:-1].gt(w.iloc[-1]).all()))


df_6['close_shifted'] = df_6['close'].shift(5)

df_6['change_pct'] = (df_6['close'] - df_6['close_shifted'])/df_6['close_shifted']*100
df_6 = df_6.drop(['close_shifted'], axis=1)

#df_6[(df_6['greater']== 1) & (df_6['change_pct']> 100)] #filter condition
df_6.tail() 
```

::: {.output .execute_result execution_count="15"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DASHUSDT</th>
      <th>SOLUSDT</th>
      <th>close</th>
      <th>six_greater</th>
      <th>six_lower</th>
      <th>change_pct</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-08-23 02:00:00</th>
      <td>47.01</td>
      <td>35.20</td>
      <td>11.81</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>18.812877</td>
    </tr>
    <tr>
      <th>2022-08-23 06:00:00</th>
      <td>46.90</td>
      <td>34.94</td>
      <td>11.96</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>13.688213</td>
    </tr>
    <tr>
      <th>2022-08-23 10:00:00</th>
      <td>47.98</td>
      <td>35.66</td>
      <td>12.32</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5.751073</td>
    </tr>
    <tr>
      <th>2022-08-23 14:00:00</th>
      <td>47.92</td>
      <td>35.69</td>
      <td>12.23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.563764</td>
    </tr>
    <tr>
      <th>2022-08-23 18:00:00</th>
      <td>47.98</td>
      <td>35.30</td>
      <td>12.68</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.457627</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
### Combined dataframe:
:::

::: {.cell .code execution_count="65"}
``` {.python}
dfs= [data, df_bb,df_atr,df_ema,df_eot,df_6]
df = pd.concat(dfs, join='outer', axis=1)#.fillna(0)
df = df.loc[:,~df.columns.duplicated()].copy()
df.tail()
```

::: {.output .execute_result execution_count="65"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DASHUSDT</th>
      <th>SOLUSDT</th>
      <th>close</th>
      <th>upper</th>
      <th>lower</th>
      <th>above_upper_bb</th>
      <th>below_lower_bb</th>
      <th>atr</th>
      <th>sell_stop_loss</th>
      <th>buy_stop_loss</th>
      <th>...</th>
      <th>white_line</th>
      <th>red_line</th>
      <th>prev_white_line</th>
      <th>prev_red_line</th>
      <th>crossover</th>
      <th>top_level_eot</th>
      <th>bottom_level_eot</th>
      <th>six_greater</th>
      <th>six_lower</th>
      <th>change_pct</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-08-23 02:00:00</th>
      <td>47.01</td>
      <td>35.20</td>
      <td>11.81</td>
      <td>11.994950</td>
      <td>8.618050</td>
      <td>False</td>
      <td>False</td>
      <td>2.597411</td>
      <td>15.706116</td>
      <td>7.913884</td>
      <td>...</td>
      <td>88.790738</td>
      <td>89.695698</td>
      <td>85.025340</td>
      <td>81.772969</td>
      <td>bearish crossover</td>
      <td>True</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>18.812877</td>
    </tr>
    <tr>
      <th>2022-08-23 06:00:00</th>
      <td>46.90</td>
      <td>34.94</td>
      <td>11.96</td>
      <td>12.172069</td>
      <td>8.738931</td>
      <td>False</td>
      <td>False</td>
      <td>2.558917</td>
      <td>15.798375</td>
      <td>8.121625</td>
      <td>...</td>
      <td>90.920390</td>
      <td>97.024559</td>
      <td>88.790738</td>
      <td>89.695698</td>
      <td>bearish crossover</td>
      <td>True</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>13.688213</td>
    </tr>
    <tr>
      <th>2022-08-23 10:00:00</th>
      <td>47.98</td>
      <td>35.66</td>
      <td>12.32</td>
      <td>12.348383</td>
      <td>8.918617</td>
      <td>False</td>
      <td>False</td>
      <td>2.546989</td>
      <td>16.140484</td>
      <td>8.499516</td>
      <td>...</td>
      <td>94.004106</td>
      <td>103.553014</td>
      <td>90.920390</td>
      <td>97.024559</td>
      <td>none</td>
      <td>True</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5.751073</td>
    </tr>
    <tr>
      <th>2022-08-23 14:00:00</th>
      <td>47.92</td>
      <td>35.69</td>
      <td>12.23</td>
      <td>12.451200</td>
      <td>9.154800</td>
      <td>False</td>
      <td>False</td>
      <td>2.535856</td>
      <td>16.033785</td>
      <td>8.426215</td>
      <td>...</td>
      <td>96.823142</td>
      <td>109.299294</td>
      <td>94.004106</td>
      <td>103.553014</td>
      <td>none</td>
      <td>True</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.563764</td>
    </tr>
    <tr>
      <th>2022-08-23 18:00:00</th>
      <td>47.98</td>
      <td>35.30</td>
      <td>12.68</td>
      <td>12.658998</td>
      <td>9.284002</td>
      <td>True</td>
      <td>False</td>
      <td>2.549466</td>
      <td>16.504199</td>
      <td>8.855801</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>96.823142</td>
      <td>109.299294</td>
      <td>none</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.457627</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="66"}
``` {.python}
df.columns
```

::: {.output .execute_result execution_count="66"}
    Index(['DASHUSDT', 'SOLUSDT', 'close', 'upper', 'lower', 'above_upper_bb',
           'below_lower_bb', 'atr', 'sell_stop_loss', 'buy_stop_loss', 'ema_200',
           'price_above_ema_200', 'price_below_ema_200', 'trig', 'white_line',
           'red_line', 'prev_white_line', 'prev_red_line', 'crossover',
           'top_level_eot', 'bottom_level_eot', 'six_greater', 'six_lower',
           'change_pct'],
          dtype='object')
:::
:::

::: {.cell .markdown}
### Sell Signals:
:::

::: {.cell .code execution_count="72" scrolled="false"}
``` {.python}
sell_df = df[ (df.above_upper_bb == True) & (df.price_above_ema_200 == True) & (df.top_level_eot == True) & (df.six_greater==1) & (df.change_pct>15) ].tail(10) 
sell_df
```

::: {.output .execute_result execution_count="72"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DASHUSDT</th>
      <th>SOLUSDT</th>
      <th>close</th>
      <th>upper</th>
      <th>lower</th>
      <th>above_upper_bb</th>
      <th>below_lower_bb</th>
      <th>atr</th>
      <th>sell_stop_loss</th>
      <th>buy_stop_loss</th>
      <th>...</th>
      <th>white_line</th>
      <th>red_line</th>
      <th>prev_white_line</th>
      <th>prev_red_line</th>
      <th>crossover</th>
      <th>top_level_eot</th>
      <th>bottom_level_eot</th>
      <th>six_greater</th>
      <th>six_lower</th>
      <th>change_pct</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-07-22 18:00:00</th>
      <td>48.02</td>
      <td>40.19</td>
      <td>7.83</td>
      <td>7.388467</td>
      <td>3.284533</td>
      <td>True</td>
      <td>False</td>
      <td>3.167040</td>
      <td>12.580561</td>
      <td>3.079439</td>
      <td>...</td>
      <td>110.000000</td>
      <td>110.000000</td>
      <td>102.606447</td>
      <td>110.000000</td>
      <td>none</td>
      <td>True</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>55.357143</td>
    </tr>
    <tr>
      <th>2022-07-22 22:00:00</th>
      <td>48.33</td>
      <td>40.49</td>
      <td>7.84</td>
      <td>7.555320</td>
      <td>3.631680</td>
      <td>True</td>
      <td>False</td>
      <td>3.181904</td>
      <td>12.612857</td>
      <td>3.067143</td>
      <td>...</td>
      <td>110.000000</td>
      <td>110.000000</td>
      <td>110.000000</td>
      <td>110.000000</td>
      <td>none</td>
      <td>True</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>35.405872</td>
    </tr>
    <tr>
      <th>2022-07-23 02:00:00</th>
      <td>49.09</td>
      <td>41.07</td>
      <td>8.02</td>
      <td>7.906902</td>
      <td>3.638098</td>
      <td>True</td>
      <td>False</td>
      <td>3.207777</td>
      <td>12.831666</td>
      <td>3.208334</td>
      <td>...</td>
      <td>109.948347</td>
      <td>110.000000</td>
      <td>110.000000</td>
      <td>110.000000</td>
      <td>none</td>
      <td>True</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>23.765432</td>
    </tr>
    <tr>
      <th>2022-07-28 18:00:00</th>
      <td>52.51</td>
      <td>42.13</td>
      <td>10.38</td>
      <td>9.839248</td>
      <td>6.730752</td>
      <td>True</td>
      <td>False</td>
      <td>1.866188</td>
      <td>13.179283</td>
      <td>7.580717</td>
      <td>...</td>
      <td>91.764054</td>
      <td>70.694224</td>
      <td>84.880802</td>
      <td>62.455452</td>
      <td>none</td>
      <td>True</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>32.736573</td>
    </tr>
    <tr>
      <th>2022-07-29 02:00:00</th>
      <td>54.35</td>
      <td>43.46</td>
      <td>10.89</td>
      <td>10.413795</td>
      <td>6.513205</td>
      <td>True</td>
      <td>False</td>
      <td>2.078546</td>
      <td>14.007820</td>
      <td>7.772180</td>
      <td>...</td>
      <td>94.916796</td>
      <td>85.895548</td>
      <td>92.502391</td>
      <td>78.542780</td>
      <td>none</td>
      <td>True</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>22.635135</td>
    </tr>
    <tr>
      <th>2022-08-17 22:00:00</th>
      <td>53.04</td>
      <td>40.51</td>
      <td>12.53</td>
      <td>11.862065</td>
      <td>8.294935</td>
      <td>True</td>
      <td>False</td>
      <td>2.203096</td>
      <td>15.834643</td>
      <td>9.225357</td>
      <td>...</td>
      <td>101.278355</td>
      <td>69.941555</td>
      <td>82.347144</td>
      <td>54.365486</td>
      <td>none</td>
      <td>True</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>26.565657</td>
    </tr>
    <tr>
      <th>2022-08-18 02:00:00</th>
      <td>53.46</td>
      <td>40.80</td>
      <td>12.66</td>
      <td>12.285692</td>
      <td>8.126308</td>
      <td>True</td>
      <td>False</td>
      <td>2.298889</td>
      <td>16.108334</td>
      <td>9.211666</td>
      <td>...</td>
      <td>106.871661</td>
      <td>83.351179</td>
      <td>101.278355</td>
      <td>69.941555</td>
      <td>none</td>
      <td>True</td>
      <td>False</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>24.852071</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 24 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
### Buy Signals:
:::

::: {.cell .code execution_count="68" scrolled="false"}
``` {.python}
buy_df = df[(df.below_lower_bb == True) & (df.price_below_ema_200 == True) & (df.bottom_level_eot == True) & (df.six_lower==1) & (df.change_pct<-50) ].tail(10) 
#(df.below_lower_bb == True) & (df.price_below_ema_200 == True) & (df.bottom_level_eot == True) & (df.six_lower==1) & (df.change_pct<80) 
buy_df
```

::: {.output .execute_result execution_count="68"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DASHUSDT</th>
      <th>SOLUSDT</th>
      <th>close</th>
      <th>upper</th>
      <th>lower</th>
      <th>above_upper_bb</th>
      <th>below_lower_bb</th>
      <th>atr</th>
      <th>sell_stop_loss</th>
      <th>buy_stop_loss</th>
      <th>...</th>
      <th>white_line</th>
      <th>red_line</th>
      <th>prev_white_line</th>
      <th>prev_red_line</th>
      <th>crossover</th>
      <th>top_level_eot</th>
      <th>bottom_level_eot</th>
      <th>six_greater</th>
      <th>six_lower</th>
      <th>change_pct</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-04-20 02:00:00</th>
      <td>107.76</td>
      <td>108.23</td>
      <td>-0.47</td>
      <td>8.453937</td>
      <td>1.223063</td>
      <td>False</td>
      <td>True</td>
      <td>4.855335</td>
      <td>6.813003</td>
      <td>-7.753003</td>
      <td>...</td>
      <td>-71.328875</td>
      <td>16.595000</td>
      <td>-56.661690</td>
      <td>27.329610</td>
      <td>none</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-109.090909</td>
    </tr>
    <tr>
      <th>2022-04-20 10:00:00</th>
      <td>109.00</td>
      <td>110.33</td>
      <td>-1.33</td>
      <td>8.670030</td>
      <td>-0.314030</td>
      <td>False</td>
      <td>True</td>
      <td>5.213581</td>
      <td>6.490372</td>
      <td>-9.150372</td>
      <td>...</td>
      <td>-77.648582</td>
      <td>-8.968412</td>
      <td>-77.201320</td>
      <td>4.269266</td>
      <td>none</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-148.540146</td>
    </tr>
    <tr>
      <th>2022-04-25 02:00:00</th>
      <td>97.61</td>
      <td>97.97</td>
      <td>-0.36</td>
      <td>3.142305</td>
      <td>-0.344305</td>
      <td>False</td>
      <td>True</td>
      <td>4.201564</td>
      <td>5.942346</td>
      <td>-6.662346</td>
      <td>...</td>
      <td>-14.100924</td>
      <td>36.166404</td>
      <td>-5.209245</td>
      <td>39.474895</td>
      <td>none</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-209.090909</td>
    </tr>
    <tr>
      <th>2022-04-25 06:00:00</th>
      <td>93.77</td>
      <td>95.09</td>
      <td>-1.32</td>
      <td>3.351483</td>
      <td>-0.717483</td>
      <td>False</td>
      <td>True</td>
      <td>4.215460</td>
      <td>5.003190</td>
      <td>-7.643190</td>
      <td>...</td>
      <td>-25.157174</td>
      <td>32.337392</td>
      <td>-14.100924</td>
      <td>36.166404</td>
      <td>none</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1750.000000</td>
    </tr>
    <tr>
      <th>2022-04-25 18:00:00</th>
      <td>99.17</td>
      <td>100.59</td>
      <td>-1.42</td>
      <td>3.263861</td>
      <td>-1.366861</td>
      <td>False</td>
      <td>True</td>
      <td>4.102513</td>
      <td>4.733770</td>
      <td>-7.573770</td>
      <td>...</td>
      <td>-20.702414</td>
      <td>23.358666</td>
      <td>-17.111097</td>
      <td>26.784631</td>
      <td>none</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-282.051282</td>
    </tr>
    <tr>
      <th>2022-04-27 06:00:00</th>
      <td>96.06</td>
      <td>99.45</td>
      <td>-3.39</td>
      <td>1.679135</td>
      <td>-3.315135</td>
      <td>False</td>
      <td>True</td>
      <td>3.846331</td>
      <td>2.379496</td>
      <td>-9.159496</td>
      <td>...</td>
      <td>-48.902009</td>
      <td>3.569222</td>
      <td>-39.446893</td>
      <td>8.190843</td>
      <td>none</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1795.000000</td>
    </tr>
    <tr>
      <th>2022-05-15 22:00:00</th>
      <td>63.04</td>
      <td>58.80</td>
      <td>4.24</td>
      <td>11.201519</td>
      <td>5.474481</td>
      <td>False</td>
      <td>True</td>
      <td>4.416562</td>
      <td>10.864844</td>
      <td>-2.384844</td>
      <td>...</td>
      <td>-90.000000</td>
      <td>-89.510126</td>
      <td>-90.000000</td>
      <td>-25.007460</td>
      <td>none</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-56.423433</td>
    </tr>
    <tr>
      <th>2022-05-16 06:00:00</th>
      <td>57.41</td>
      <td>53.78</td>
      <td>3.63</td>
      <td>11.393101</td>
      <td>4.197899</td>
      <td>False</td>
      <td>True</td>
      <td>4.625228</td>
      <td>10.567842</td>
      <td>-3.307842</td>
      <td>...</td>
      <td>-90.000000</td>
      <td>-90.000000</td>
      <td>-90.000000</td>
      <td>-90.000000</td>
      <td>none</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-58.983051</td>
    </tr>
    <tr>
      <th>2022-05-16 10:00:00</th>
      <td>56.97</td>
      <td>53.78</td>
      <td>3.19</td>
      <td>11.439640</td>
      <td>3.502360</td>
      <td>False</td>
      <td>True</td>
      <td>4.768213</td>
      <td>10.342319</td>
      <td>-3.962319</td>
      <td>...</td>
      <td>-88.855605</td>
      <td>-90.000000</td>
      <td>-90.000000</td>
      <td>-90.000000</td>
      <td>none</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-59.974906</td>
    </tr>
    <tr>
      <th>2022-07-19 14:00:00</th>
      <td>49.24</td>
      <td>46.54</td>
      <td>2.70</td>
      <td>8.177263</td>
      <td>3.271737</td>
      <td>False</td>
      <td>True</td>
      <td>2.613543</td>
      <td>6.620314</td>
      <td>-1.220314</td>
      <td>...</td>
      <td>-86.202331</td>
      <td>8.250075</td>
      <td>-90.000000</td>
      <td>43.547199</td>
      <td>none</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-59.641256</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 24 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
### Visualizing:
:::

::: {.cell .code execution_count="73"}
``` {.python}
figure(figsize=(12,6), dpi=200)
df= df.tail(300)
buy_df = buy_df.tail(1)
sell_df = sell_df.tail(1)
plt.plot(df.close, label=f'Price pair ({coin1}-{coin2})')
plt.plot(df.upper, label=f'BB-Upper')
plt.plot(df.lower, label=f'BB-Lower')
plt.plot(df.ema_200, label=f'EMA-200')
plt.scatter(buy_df.index, buy_df.close.to_numpy(), marker ='^', s=150, c='g')
plt.scatter(sell_df.index, sell_df.close.to_numpy(), marker ='v', s=150, c='r')
plt.legend()
```

::: {.output .execute_result execution_count="73"}
    <matplotlib.legend.Legend at 0x7f8e9d4c9280>
:::

::: {.output .display_data}
![](vertopal_51c2f706b6004e63954e71c5561a235c/d9530ac0a0feebe3bd86587520ffdc01819fb539.png)
:::
:::

::: {.cell .markdown}
### Commit
:::

::: {.cell .code}
``` {.python}
# Execute this to save new versions of the notebook
jovian.commit(project="ranton-mean-reversion-study-1")
```

::: {.output .display_data}
    <IPython.core.display.Javascript object>
:::
:::

::: {.cell .code}
``` {.python}
```
:::
