#!/usr/bin/env python
# coding: utf-8

# In[1]:

from tqdm import trange
import pickle
import os
import shutil

from agent2 import ValidationAgent
from function import getStockDataVec, getState, formatPrice


# In[7]:


def no_greedy(stockname, mode):
    window_size = 20
    record = []


    for ii in trange(1,51):
        agent = ValidationAgent(window_size,ii,True)
        #print(agent.policy_net)
        data = getStockDataVec(stockname, mode)
        dataclose = data[1]
        dataopen = data[0]
        l = len(dataclose)
        #batch_size = 32
        state = getState(data, window_size + 1, window_size + 1)
        total_profit = 0
        agent.inventory = []
        buys = window_size*[None]
        sells = window_size*[None]
        capital = 100000
        tradenum = 0
        actionlist = (window_size - 1)*[5]

        for t in range(window_size, l):
            reward = 0
            action = agent.act(state) 
            actionlist.append(action)

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
                    reward = dataopen[t] - bought_price - 2 #手續費
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

            next_state = getState(data, t + 1, window_size + 1)
            done = True if t == l - 1 else False
            agent.memory.push(state, action, next_state, reward)
            state = next_state

            if done:
#                 print("--------------------------------")
#                 print(f'第{ii}次')
#                 print(stockname + " Total Profit: " + formatPrice(total_profit))
#                 print('交易次數 : ' + str(tradenum))
#                 print("--------------------------------")
#                 print(actionlist.count(0),actionlist.count(1))
                record.append((total_profit,tradenum,ii))
    return record



# In[9]:


def bestepoch(recordtrain,recordtest,num,mode,stockname):
    #最優
    train = sorted(recordtrain, key = lambda a : a[0], reverse = True)
    test = sorted(recordtest, key = lambda a : a[0], reverse = True)
    trainsort = [train[i][2] for i in range(len(train))]
    testsort = [test[i][2] for i in range(len(test))]
    bestepoch = [same for same in testsort[:30] if same in trainsort[:30]]

    window_size = 20
    ii = bestepoch[0]

    agent = ValidationAgent(window_size,ii,True)#,True
    #print(agent.policy_net)
    data = getStockDataVec(stockname, 'all')
    dataclose = data[1]
    dataopen = data[0]
    l = len(dataclose)
    #batch_size = 32
    state = getState(data, window_size + 1, window_size + 1)
    total_profit = 0
    agent.inventory = []
    buys = window_size*[None]
    sells = window_size*[None]
    capital = 100000
    tradenum = 0
    actionlist = (window_size - 1)*[5]
    
    #保存最好的模型
    if not os.path.isdir(f'./best_models/{stockname}'):
        os.mkdir(f'./best_models/{stockname}')
    shutil.move(f'./models/policy_model_{ii}',f'./best_models/{stockname}')
    shutil.move(f'./models/target_model_{ii}',f'./best_models/{stockname}')
    os.rename(f'./best_models/{stockname}/policy_model_{ii}', f'./best_models/{stockname}/policy_model_{mode}_{num}_{ii}')
    os.rename(f'./best_models/{stockname}/target_model_{ii}', f'./best_models/{stockname}/target_model_{mode}_{num}_{ii}')

    for t in trange(window_size, l):
        reward = 0
        action = agent.act(state) 
        actionlist.append(action)

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
                reward = dataopen[t] - bought_price - 2 #手續費
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


        next_state = getState(data, t + 1, window_size + 1)
        done = True if t == l - 1 else False
        agent.memory.push(state, action, next_state, reward)
        state = next_state


        if done:
            print("--------------------------------")
            print(f'第{ii}次')
            print(stockname + " Total Profit: " + formatPrice(total_profit))
            print('交易次數 : ' + str(tradenum))
            print("--------------------------------")
            print(actionlist.count(0),actionlist.count(1))
#             sss = {
#                 'closes': Series(dataclose),
#                 'action': Series(actionlist),
#                 'buys' : Series(buys),
#                 'sells': Series(sells)
#             }

            #儲存變數
            def save_variable(filename,v):
                f=open(filename,'wb')
                pickle.dump(v,f)
                f.close()
                return filename
            x_data = range(len(dataclose))
            save_variable('x_data', x_data)
            save_variable('closes', dataclose)
            save_variable('buys', buys)
            save_variable('sells', sells)
            
            return [total_profit,tradenum,ii]
    

