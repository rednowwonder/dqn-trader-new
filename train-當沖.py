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
data = getStockDataVec('DJI_2007','train')
dataclose = data[1]
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
    ax2.plot(x_data, totalreturnlist, color='black', alpha=1)
    ax2.tick_params(axis='y', labelcolor='black')

    fig.tight_layout()
    plt.savefig(f'./reward/total_return_{mode}_{num}')
    plt.show()
    
    totalreward = 0
    for i in range(math.ceil(len(sss)*0.8),len(sss)):
        if sss.loc[i-1,'action'] == 0:
            totalreward += sss.loc[i,'sells'] - sss.loc[i,'buys'] -2
        if sss.loc[i-1,'action'] == 1:
            totalreward += sss.loc[i,'buys'] - sss.loc[i,'sells'] -2
    print('test 時間段的報酬 : '+ str(totalreward))
    indexreturn = closes[len(closes)-1] - closes[math.ceil(len(closes)*0.8)-1]
    print('test 時間段的指數報酬 :' + str(indexreturn))


# In[ ]:





# In[ ]:




