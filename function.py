#!/usr/bin/env python
# coding: utf-8

# In[36]:

import numpy as np
import math

# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file

def getStockDataVec(key,mode):
    closes = []
    opens = []
    highs = []
    lows = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()
    
    for line in lines[1:]:
        close = line.split(",")[4]
        if close != 'null':
            closes.append(float(line.split(",")[4]))
        openp = line.split(",")[1]
        if openp != 'null':
            opens.append(float(line.split(",")[1]))
        high = line.split(",")[2]
        if high != 'null':
            highs.append(float(line.split(",")[2]))
        low = line.split(",")[3]
        if low != 'null':
            lows.append(float(line.split(",")[3]))
    if mode == 'train':
        closes = closes[:math.ceil(len(closes)*0.6)]
        opens = opens[:math.ceil(len(opens)*0.6)]
        #highs = highs[:math.ceil(len(highs)*0.6)]
        #lows = lows[:math.ceil(len(lows)*0.6)]
    if mode == 'valid':
        closes = closes[math.ceil(len(closes)*0.6):math.ceil(len(closes)*0.8)]
        opens = opens[math.ceil(len(opens)*0.6):math.ceil(len(opens)*0.8)]
        #highs = highs[math.ceil(len(highs)*0.6):math.ceil(len(highs)*0.8)]
        #lows = lows[math.ceil(len(lows)*0.6):math.ceil(len(lows)*0.8)]
    if mode == 'test':
        closes = closes[math.ceil(len(closes)*0.8):]
        opens = opens[math.ceil(len(opens)*0.8):]
        #highs = highs[math.ceil(len(highs)*0.8):]
        #lows = lows[math.ceil(len(lows)*0.8):]
    if mode == 'all':
        return  [opens, closes]#, highs, lows
    return [opens, closes]#, highs, lows

# returns the sigmoid
def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x)/(1 + np.exp(x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
    dataopen = data[0]
    #datahigh = getStockDataVec(stockname, mode)[1]
    #datalow = getStockDataVec(stockname, mode)[2]
    dataclose = data[1]
    #d = t - n + 1
    #block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    
    blockopen = dataopen[t-n:t]
    #blockhigh = datahigh[t-n:t]
    #blocklow = datalow[t-n:t]
    blockclose = dataclose[t-n:t]
    resopen,reshigh,reslow,resclose = [],[],[],[]
    for i in range(n-1):
        resopen.append(sigmoid(blockopen[i + 1] - blockopen[i]))
        #reshigh.append(sigmoid(blockhigh[i + 1] - blockhigh[i]))
        #reslow.append(sigmoid(blocklow[i + 1] - blocklow[i]))
        resclose.append(sigmoid(blockclose[i + 1] - blockclose[i]))

    #return np.array([resopen,reshigh,reslow,resclose])
    #return np.array([blockopen[:-1],blockhigh[:-1],blocklow[:-1],blockclose[:-1]])
    return np.array([resopen,resclose])
    #return np.array([resclose])


# In[ ]:


