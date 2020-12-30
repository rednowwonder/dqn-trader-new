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


def plot_function(num,mode):
    def load_variavle(filename):
        f=open(filename,'rb')
        r=pickle.load(f)
        f.close()
        return r
    
    if not os.path.isdir('./reward'):
        os.mkdir('./reward')

    closes = load_variavle('closes')
    x_data = range(len(closes))
    buys = load_variavle('buys')
    sells = load_variavle('sells')
    
    sss = {
    'closes': Series(closes),
    'buys' : Series(buys),
    'sells': Series(sells)
    }
    sss = pd.concat([DataFrame(sss).iloc[:,:2].fillna(method='pad',axis=0),DataFrame(sss).iloc[:,-1]],axis=1)
    sssnull = sss.isnull()
    
    totalrewardlist = []
    totalreward = 0
    for i in range(len(sss)):
        reward = 0
        if sssnull.iloc[i,1] == True and sssnull.iloc[i,2] == True:
            totalreward += 0
        if sssnull.iloc[i,1] != True and sssnull.iloc[i,2] == True:
            reward = sss.iloc[i,0] - sss.iloc[i,1]
        if sssnull.iloc[i,1] != True and sssnull.iloc[i,2] != True:
            totalreward += sss.iloc[i,2] - sss.iloc[i,1] - 2
        totalrewardlist.append(totalreward + reward)
    
    fig, ax1 = plt.subplots()
    plt.title('Total Return')
    plt.xlabel('Time')
    ax2 = ax1.twinx()

    ax1.set_ylabel('DJI index', color='tab:blue')
    plt.axvline(x=math.ceil(len(closes)*0.6), ymin=0, ymax=1, color = 'red')
    plt.axvline(x=math.ceil(len(closes)*0.8), ymin=0, ymax=1, color = 'red')
    ax1.plot(x_data, closes, color='tab:blue', alpha=0.75)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2.set_ylabel('Accumulated return', color='black')
    ax2.plot(x_data, totalrewardlist, color='black', alpha=1)
    ax2.tick_params(axis='y', labelcolor='black')

    fig.tight_layout()
    plt.savefig(f'./reward/total_return_{mode}_{num}')
    plt.show()
    
    
    totalreturn = 0
    for i in range(math.ceil(len(sssnull)*0.8),len(sssnull)):
        reward = 0
        if sssnull.iloc[i,1] == True and sssnull.iloc[i,2] == True:
            totalreturn += 0
        if sssnull.iloc[i,1] != True and sssnull.iloc[i,2] == True:
            reward = sss.iloc[i,0] - sss.iloc[i,1]
        if sssnull.iloc[i,1] != True and sssnull.iloc[i,2] != True:
            totalreturn += sss.iloc[i,2] - sss.iloc[i,1] - 2
    print('test 時間段的報酬 : '+ str(totalreturn))
    indexreturn = closes[len(closes)-1] - closes[math.ceil(len(closes)*0.8)-1]
    print('test 時間段的指數報酬 :' + str(indexreturn))


# In[ ]:




