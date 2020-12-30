#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from pandas import Series,DataFrame
from matplotlib import pyplot as plt
from agent2 import ValidationAgent
from function import getStockDataVec, getState, formatPrice
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# In[2]:


stock_name = '^N225'
window_size = 20
recordtrain = []


for ii in range(1,101):
    agent = ValidationAgent(window_size,ii,True)
    #print(agent.policy_net)
    dataclose = getStockDataVec(stock_name,900,1)[0]
    dataopen = getStockDataVec(stock_name,900,1)[1]
    l = len(dataclose)
    #batch_size = 32
    state = getState(dataclose, window_size + 1, window_size + 1)
    total_profit = 0
    agent.inventory = []
    tradenum = 0
    actionlist = window_size*[5] + [agent.act(state)]

    for t in range(window_size + 1,l):
        reward = 0

        if actionlist[t-1] == 0: # buy
            reward = dataclose[t] - dataopen[t] -1
            total_profit += reward

        elif actionlist[t-1] == 1: # sell
            reward = dataopen[t] - dataclose[t] -1 #手續費
            total_profit += reward
            tradenum += 1
        
        action = agent.act(state) 
        actionlist.append(action)
        next_state = getState(dataclose, t + 1, window_size + 1)
        done = True if t == l - 1 else False
        agent.memory.push(state, action, next_state, reward)
        state = next_state

        if done:
            print("--------------------------------")
            print(f'第{ii}次')
            print(stock_name + " Total Profit: " + formatPrice(total_profit))
            print('交易次數 : ' + str(tradenum))
            print("--------------------------------")
            recordtrain.append((total_profit,tradenum,ii))
            print(actionlist.count(0),actionlist.count(1))

# In[3]:


stock_name = '^N225'
window_size = 20
recordtest = []

for ii in range(1,101):
    agent = ValidationAgent(window_size,ii,True)
    #print(agent.policy_net)
    dataclose = getStockDataVec(stock_name,900,0)[0]
    dataopen = getStockDataVec(stock_name,900,0)[1]

    l = len(dataclose)
    #batch_size = 32
    state = getState(dataclose, window_size + 1, window_size + 1)
    total_profit = 0
    agent.inventory = []
    capital = 100000
    tradenum = 0
    actionlist = window_size*[None] + [agent.act(state)]

    for t in range(window_size + 1,l):
        reward = 0

        if actionlist[t-1] == 0: # buy
            reward = dataclose[t] - dataopen[t] - 1
            total_profit += reward

        elif actionlist[t-1] == 1: # sell
            reward = dataopen[t] - dataclose[t] - 1 #手續費
            total_profit += reward
            tradenum += 1
        
        action = agent.act(state) 
        actionlist.append(action)
        next_state = getState(dataclose, t + 1, window_size + 1)
        done = True if t == l - 1 else False
        agent.memory.push(state, action, next_state, reward)
        state = next_state

        if done:
            print("--------------------------------")
            print(f'第{ii}次')
            print(stock_name + " Total Profit: " + formatPrice(total_profit))
            print('交易次數 : ' + str(tradenum))
            print("--------------------------------")
            recordtest.append((total_profit,tradenum,ii))
            print(actionlist.count(0),actionlist.count(1))


# In[4]:


train = sorted(recordtrain, key = lambda a : a[0], reverse = True)
test = sorted(recordtest, key = lambda a : a[0], reverse = True)
print(test)


# In[5]:


trainsort = [train[i][2] for i in range(len(train))]
testsort = [test[i][2] for i in range(len(test))]
print([same for same in testsort[:30] if same in trainsort[:30]])


# In[6]:


#最優
stock_name = '^N225'
window_size = 20
ii = 18

agent = ValidationAgent(window_size,ii,True)#,True
#print(agent.policy_net)
dataclose = getStockDataVec(stock_name,1,0)[0]
dataopen = getStockDataVec(stock_name,1,0)[1]
l = len(dataclose)
#batch_size = 32
state = getState(dataclose, window_size + 1, window_size + 1)
total_profit = 0
agent.inventory = []
buys = (window_size + 1)*[None]
sells = (window_size + 1)*[None]
capital = 100000
tradenum = 0
actionlist = window_size*[5] + [agent.act(state)]

for t in range(window_size + 1, l):
    reward = 0

    if actionlist[t-1] == 0: # buy
        buys.append(dataopen[t])
        sells.append(dataclose[t])
        reward = dataclose[t] - dataopen[t] -1
        total_profit += reward

    elif actionlist[t-1] == 1: # sell
        buys.append(dataopen[t])
        sells.append(dataclose[t])
        reward = dataopen[t] - dataclose[t] -1 #手續費
        total_profit += reward
        tradenum += 1
    else:
        buys.append(None)
        sells.append(None)

    action = agent.act(state) 
    actionlist.append(action)
    next_state = getState(dataclose, t + 1, window_size + 1)
    done = True if t == l - 1 else False
    agent.memory.push(state, action, next_state, reward)
    state = next_state


    if done:
        print("--------------------------------")
        print(f'第{ii}次')
        print(stock_name + " Total Profit: " + formatPrice(total_profit))
        print('交易次數 : ' + str(tradenum))
        print("--------------------------------")
        print(actionlist.count(0),actionlist.count(1))
        sss = {
            'closes': Series(dataclose),
            'action': Series(actionlist),
            'buys' : Series(buys),
            'sells': Series(sells)
        }


# In[ ]:


import pickle

def save_variable(filename,v):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

x_data = range(l)
# print(len(x_data))
# print(closes)
# print(buys)
# print(sells)

save_variable('x_data', x_data)
save_variable('closes', dataclose)
save_variable('action',actionlist)
save_variable('buys', buys)
save_variable('sells', sells)


# In[7]:


sss = DataFrame(sss)
sss


# In[11]:


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
        totalreturn += sss.loc[i,'sells'] - sss.loc[i,'buys'] -1
    if sss.loc[i-1,'action'] == 1:
        sellin.append(sss.loc[i,'buys'])
        buyout.append(sss.loc[i,'sells'])
        buyin.append(None)
        sellout.append(None)
        totalreturn += sss.loc[i,'buys'] - sss.loc[i,'sells'] -1
    if sss.loc[i-1,'action'] == 5:
        sellin.append(None)
        buyout.append(None)
        buyin.append(None)
        sellout.append(None)
    totalreturnlist.append(totalreturn)


# In[10]:


len(sss)


# In[ ]:


start,end = 1,300
dataclose = dataclose[start:end]
x_data = range(len(dataclose))
buyin = buyin[start:end]
buyout = buyout[start:end]
sellin = sellin[start:end]
sellout = sellout[start:end]

plt.figure(figsize=(20, 10))
plt.plot(x_data, dataclose)
plt.scatter(x_data, buyin, marker='^', s=32, c='r')
plt.scatter(x_data, buyout, marker='^', s=32, c='g')
plt.scatter(x_data, sellin, marker='v', s=32, c='r')
plt.scatter(x_data, sellout, marker='v', s=32, c='g')


# In[12]:


x = range(len(totalreturnlist))
plt.plot(x,totalreturnlist)

