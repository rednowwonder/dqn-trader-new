#!/usr/bin/env python
# coding: utf-8

# In[1]:

from agent2 import TrainAgent
from function import getStockDataVec, getState, formatPrice
import torch


# In[2]:

window_size = 20
episode_count = 100
agent = TrainAgent(window_size)
dataclose = getStockDataVec('DJI_F','train')[3]
dataopen = getStockDataVec('DJI_F','train')[0]
l = len(dataclose)

for e in range(episode_count):
    print("Episode " + str(e+1) + "/" + str(episode_count))
    state = getState('DJI_F','train', window_size + 1, window_size + 1)
    total_profit = 0
    agent.inventory = []
    actionlist = window_size*[None]
    
    for t in range(window_size, l):
        reward = 0
        action = agent.act(state)
        actionlist.append(action)
        if actionlist[t-1] == 0: # buy
            reward = dataclose[t] - dataopen[t] -1
            total_profit += reward
            # print("Buy: " + formatPrice(dataopen[t]))

        elif actionlist[t-1] == 1: # sell
            reward = dataopen[t] - dataclose[t] -1 #手續費
            total_profit += reward
            # print("Sell: " + formatPrice(dataopen[t]) + " | Profit: " + formatPrice(dataopen[t] - bought_price))
        # print(agent.inventory,action)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon = agent.epsilon * agent.epsilon_decay
        else:
            agent.epsilon = agent.epsilon_min
        next_state = getState('DJI_F','train', t + 1 , window_size + 1)
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




