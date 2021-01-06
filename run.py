#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',10)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# In[2]:

from train_function_只做多一口 import train_function
from validation_function_只做多一口 import no_greedy,bestepoch
from plot_function_只做多一口 import plot_function
from train_function_當沖 import train_function_daytrade
from validation_function_當沖 import no_greedy_daytrade,bestepoch_daytrade
from plot_function_當沖 import plot_function_daytrade

stockname = 'DJI_2007'
# In[3]:


#用只做多一口訓練且用只做多一口當實際交易策略

totalreturn1 = []
tradenum1 = []
bestepoch1 = []
for i in range(5):
    train_function(stockname,'train')
    recordtrain = no_greedy(stockname,'train')
    recordtest = no_greedy(stockname,'valid')
    out = bestepoch(recordtrain,recordtest,i,1,stockname)
    totalreturn1.append(out[0])
    tradenum1.append(out[1])
    bestepoch1.append(out[2])
    plot_function(i,1,stockname)
    print('-----------')


# In[ ]:


#用只做多一口訓練且用當沖當實際交易策略

totalreturn2 = []
tradenum2 = []
bestepoch2 = []
for i in range(5):
    train_function(stockname,'train')
    recordtrain = no_greedy_daytrade(stockname,'train')
    recordtest = no_greedy_daytrade(stockname,'valid')
    out = bestepoch_daytrade(recordtrain,recordtest,i,2,stockname)
    totalreturn2.append(out[0])
    tradenum2.append(out[1])
    bestepoch2.append(out[2])
    plot_function_daytrade(i,2,stockname)
    print('-----------')


# In[ ]:


#用當沖訓練且用當沖當實際交易策略

totalreturn3 = []
tradenum3 = []
bestepoch3 = []
for i in range(5):
    train_function_daytrade(stockname,'train')
    recordtrain = no_greedy_daytrade(stockname,'train')
    recordtest = no_greedy_daytrade(stockname,'valid')
    out = bestepoch_daytrade(recordtrain,recordtest,i,3,stockname)
    totalreturn3.append(out[0])
    tradenum3.append(out[1])
    bestepoch3.append(out[2])
    plot_function_daytrade(i,3,stockname)
    print('-----------')


# In[ ]:


#用當沖訓練且用只做多一口當實際交易策略

totalreturn4 = []
tradenum4 = []
bestepoch4 = []
for i in range(5):
    train_function_daytrade(stockname,'train')
    recordtrain = no_greedy(stockname,'train')
    recordtest = no_greedy(stockname,'valid')
    out = bestepoch(recordtrain,recordtest,i,4,stockname)
    totalreturn4.append(out[0])
    tradenum4.append(out[1])
    bestepoch4.append(out[2])
    plot_function(i,4,stockname)
    print('-----------')

# In[ ]:


totalreturnmean = [np.mean(totalreturn1),np.mean(totalreturn2),np.mean(totalreturn3),np.mean(totalreturn4)]
tradenummean = [np.mean(tradenum1),np.mean(tradenum2),np.mean(tradenum3),np.mean(tradenum4)]
print(pd.DataFrame({
    '平均總損益':totalreturnmean,
    '平均交易次數':tradenummean
}))

# In[ ]:


print(pd.DataFrame({
    '總損益1':totalreturn1,
    '交易次數1':tradenum1,
    '最佳epoch1':bestepoch1,
    '總損益2':totalreturn2,
    '交易次數2':tradenum2,
    '最佳epoch2':bestepoch2,
    '總損益3':totalreturn3,
    '交易次數3':tradenum3,
    '最佳epoch3':bestepoch3,
    '總損益4':totalreturn4,
    '交易次數4':tradenum4,
    '最佳epoch4':bestepoch4
}))


# In[ ]:




