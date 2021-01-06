#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pickle
from matplotlib import pyplot as plt
from pandas import Series,DataFrame
import pandas as pd
import os
import math


# In[18]:


def plot_function_daytrade(num,mode,key):
    def load_variable(filename):
        f=open(filename,'rb')
        r=pickle.load(f)
        f.close()
        return r
            
    if not os.path.isdir('./reward'):
        os.mkdir('./reward')
    x_data = load_variable('x_data')
    closes = load_variable('closes')
    buys = load_variable('buys')
    sells = load_variable('sells')
    action = load_variable('action')
    
    dateid = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        close = line.split(",")[4]
        if close != 'null':
            dateid.append(line.split(",")[0])
    
    sss = {
        'closes': Series(closes),
        'action': Series(action),
        'buys' : Series(buys),
        'sells': Series(sells)
    }
    sss = DataFrame(sss)

    buyin = [None]
    sellin = [None]
    buyout = [None]
    sellout = [None]
    totalreturn = 0
    totalreturnlist = [0]
    for i in range(1,len(sss)):
        if sss.loc[i-1,'action'] == 0:
            buyin.append(sss.loc[i,'buys'])
            sellout.append(sss.loc[i,'sells'])
            sellin.append(None)
            buyout.append(None)
            totalreturn += sss.loc[i,'sells'] - sss.loc[i,'buys'] -2
        if sss.loc[i-1,'action'] == 1:
            sellin.append(sss.loc[i,'buys'])
            buyout.append(sss.loc[i,'sells'])
            buyin.append(None)
            sellout.append(None)
            totalreturn += sss.loc[i,'buys'] - sss.loc[i,'sells'] -2
        if sss.loc[i-1,'action'] == 5:
            sellin.append(None)
            buyout.append(None)
            buyin.append(None)
            sellout.append(None)
        totalreturnlist.append(totalreturn)
        
    df =  DataFrame({'dateid' : dateid,'closes' : closes,'totalreturn' : totalreturnlist})
    df = df.set_index('dateid')
    df['closes'].plot(grid=1,figsize=(12,9),title=f'{key} index')
    df['totalreturn'].plot(grid=1,figsize=(12,9))
    plt.axvline(x=math.ceil(len(closes)*0.6), ymin=0, ymax=1, color = 'red')
    plt.axvline(x=math.ceil(len(closes)*0.8), ymin=0, ymax=1, color = 'red')
    plt.savefig(f'./reward/total_return_{mode}_{num}')
    plt.show()
    
    #MDD
    
    #sharpe ratio


    totalreward = 0
    for i in range(math.ceil(len(sss)*0.8),len(sss)):
        if sss.loc[i-1,'action'] == 0:
            totalreward += sss.loc[i,'sells'] - sss.loc[i,'buys'] -2
        if sss.loc[i-1,'action'] == 1:
            totalreward += sss.loc[i,'buys'] - sss.loc[i,'sells'] -2
    print('test 時間段的報酬 : '+ str(totalreward))
    indexreturn = closes[len(closes)-1] - closes[math.ceil(len(closes)*0.8)-1]
    print('test 時間段的指數報酬 :' + str(indexreturn))
    
    print(df.iloc[len(df)-1,'totalreturn']-df.iloc[math.ceil(len(df)*0.8),'totalreturn'])
    print(df.iloc[len(df)-1,'closes']-df.iloc[math.ceil(len(df)*0.8),'closes'])



# In[ ]:


