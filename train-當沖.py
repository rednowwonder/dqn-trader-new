#!/usr/bin/env python
# coding: utf-8

# In[1]:

from agent2 import TrainAgent
from function import getStockDataVec, getState, formatPrice
import torch


# In[2]:

window_size = 20
episode_count = 500
agent = TrainAgent(window_size)
data = getStockDataVec('DJI_2007','train')
dataclose = data[3]
dataopen = data[0]
l = len(dataclose)
buys = window_size*[None]
sells = window_size*[None]
tradenum = 0

for e in range(episode_count):
    print("Episode " + str(e+1) + "/" + str(episode_count))
    state = getState(data, window_size + 1, window_size + 1)
    total_profit = 0
    agent.inventory = []
    actionlist = window_size*[None]
    agent.epsilon = 1
    
    for t in range(window_size, l):
        reward = 0
        action = agent.act(state)
        actionlist.append(action)
        if actionlist[t-1] == 0: # buy
            buys.append(dataopen[t])
            sells.append(dataclose[t])
            reward = dataclose[t] - dataopen[t] -2
            total_profit += reward
            tradenum += 1
            # print("Buy: " + formatPrice(dataopen[t]))

        elif actionlist[t-1] == 1: # sell
            buys.append(dataopen[t])
            sells.append(dataclose[t])
            reward = dataopen[t] - dataclose[t] -2 #手續費
            total_profit += reward
            tradenum += 1
            # print("Sell: " + formatPrice(dataopen[t]) + " | Profit: " + formatPrice(dataopen[t] - bought_price))
        # print(agent.inventory,action)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon = agent.epsilon * agent.epsilon_decay
        else:
            agent.epsilon = agent.epsilon_min
        #print(agent.epsilon)
        next_state = getState(data, t + 1 , window_size + 1)
        done = True if t == l - 1 else False
        agent.memory.push(state, action, next_state, reward)
        state = next_state
        
        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
        agent.optimize()
    print(actionlist.count(0),actionlist.count(1))#,actionlist.count(2)
    if e % 1 == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        torch.save(agent.policy_net, "models/policy_model_%d"%(e+1))
        torch.save(agent.target_net, "models/target_model_%d"%(e+1))
        
    


# In[ ]:





# In[ ]:




