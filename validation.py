#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import os
import shutil

from agent2 import ValidationAgent
from function import getStockDataVec, getState, formatPrice


# In[ ]:


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
    buys = (window_size + 1)*[None]
    sells = (window_size + 1)*[None]
    capital = 100000
    tradenum = 0
    actionlist = window_size*[5] + [agent.act(state)]

    for t in range(window_size + 1,l):
        reward = 0

        if actionlist[t-1] == 0 and len(agent.inventory) == 0: # buy
            #print("Buy: " + formatPrice(dataopen[t]))
            if capital > dataopen[t]:
                agent.inventory.append(dataopen[t])
                buys.append(dataopen[t])
                sells.append(None)
                capital -= dataopen[t]
            else:
                buys.append(None)
                sells.append(None)

        elif actionlist[t-1] == 1: # sell
            if len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                reward = dataopen[t] - bought_price - 0.2 #手續費
                total_profit += reward
                buys.append(None)
                sells.append(dataopen[t])
                capital += dataopen[t]
                #print("Sell: " + formatPrice(dataopen[t]) + " | Profit: " + formatPrice(dataopen[t] - bought_price))
                tradenum += 1
            else:
                buys.append(None)
                sells.append(None)
        elif actionlist[t-1] == 0:
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
            recordtrain.append((total_profit,tradenum,ii))
            print(actionlist.count(0),actionlist.count(1))


# In[ ]:


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
    buys = (window_size + 1)*[None]
    sells = (window_size + 1)*[None]
    capital = 100000
    tradenum = 0
    actionlist = window_size*[None] + [agent.act(state)]

    for t in range(window_size + 1, l):
        reward = 0

        #print(next_state)
        if actionlist[t-1] == 0 and len(agent.inventory) == 0: # buy
            #print("Buy: " + formatPrice(dataopen[t]))
            if capital > dataopen[t]:
                agent.inventory.append(dataopen[t])
                buys.append(dataopen[t])
                sells.append(None)
                capital -= dataopen[t]
            else:
                buys.append(None)
                sells.append(None)

        elif actionlist[t-1] == 1: # sell
            if len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                reward = dataopen[t] - bought_price - 0.2 #手續費
                total_profit += reward
                buys.append(None)
                sells.append(dataopen[t])
                capital += dataopen[t]
                #print("Sell: " + formatPrice(dataopen[t]) + " | Profit: " + formatPrice(dataopen[t] - bought_price))
                tradenum += 1
            else:
                buys.append(None)
                sells.append(None)
        elif actionlist[t-1] == 0:
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
            recordtest.append((total_profit,tradenum,ii))
            print(actionlist.count(0),actionlist.count(1))
            


# In[ ]:


train = sorted(recordtrain, key = lambda a : a[0], reverse = True)
test = sorted(recordtest, key = lambda a : a[0], reverse = True)
print(test)


# In[ ]:


trainsort = [train[i][2] for i in range(len(train))]
testsort = [test[i][2] for i in range(len(test))]
print([same for same in testsort[:30] if same in trainsort[:30]])


# In[ ]:


#最優
stock_name = '^N225'
window_size = 20
ii =48

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

    if actionlist[t-1] == 0 and len(agent.inventory) == 0: # buy
        #print("Buy: " + formatPrice(dataopen[t]))
        if capital > dataopen[t]:
            agent.inventory.append(dataopen[t])
            buys.append(dataopen[t])
            sells.append(None)
            capital -= dataopen[t]
        else:
            buys.append(None)
            sells.append(None)

    elif actionlist[t-1] == 1: # sell
        if len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = dataopen[t] - bought_price - 0.2 #手續費
            total_profit += reward
            buys.append(None)
            sells.append(dataopen[t])
            capital += dataopen[t]
            #print("Sell: " + formatPrice(dataopen[t]) + " | Profit: " + formatPrice(dataopen[t] - bought_price))
            tradenum += 1
        else:
            buys.append(None)
            sells.append(None)
    elif actionlist[t-1] == 0:
        buys.append(None)
        sells.append(None)
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


DataFrame(sss).iloc[1200:,]


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
save_variable('buys', buys)
save_variable('sells', sells)


# In[ ]:




