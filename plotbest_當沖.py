# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:48:59 2021
@author: user
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
#from tqdm import trange
import math
from matplotlib import pyplot as plt
from agent2 import ValidationAgent
from function import getStockDataVec, getState, formatPrice

window_size = 20
stockname = 'DJI_2007'
agent = ValidationAgent(window_size,220,True)#,True
#print(agent.policy_net)
data = getStockDataVec(stockname,'all')
dataclose = data[3]
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

dateid = []
lines = open("data/" + stockname + ".csv", "r").read().splitlines()
for line in lines[1:]:
    close = line.split(",")[4]
    if close != 'null':
        dateid.append(line.split(",")[0])

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

    elif actionlist[t-1] == 1: # sell
        buys.append(dataopen[t])
        sells.append(dataclose[t])
        reward = dataopen[t] - dataclose[t] -2 #手續費
        total_profit += reward
        tradenum += 1
    elif actionlist[t-1] == 5:
        buys.append(None)
        sells.append(None)

    next_state = getState(data, t + 1, window_size + 1)
    done = True if t == l - 1 else False
    agent.memory.push(state, action, next_state, reward)
    state = next_state


    if done:
        print("--------------------------------")
        print(f'第{220}次')
        print(stockname + " Total Profit: " + formatPrice(total_profit))
        print('交易次數 : ' + str(tradenum))
        print("--------------------------------")
        print(actionlist.count(0),actionlist.count(1))
        actionlist.append(agent.act(state))
        sss = {'date' : dateid,
              'closes': Series(dataclose),
                 'action': Series(actionlist),
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
    
df =  DataFrame({'date' : dateid,'DJI_closes' : dataclose,'totalreturn' : totalreturnlist})
df = df.set_index('date')
df.plot(grid=1,figsize=(12,9),title='Dow Jones Index')
#df['closes'].plot(grid=1,figsize=(12,9),title=f'{stockname} index')
#df['totalreturn'].plot(grid=1,figsize=(12,9))
plt.axvline(x=math.ceil(len(dataclose)*0.6), ymin=0, ymax=1, color = 'red')
plt.axvline(x=math.ceil(len(dataclose)*0.8), ymin=0, ymax=1, color = 'red')
plt.show()

actsignal = DataFrame({'date':dateid,'actsignal':actionlist})
actsignal = actsignal.set_index('date')
actsignal.plot(grid = 1,figsize = (12,9),title = 'signal')

fig, axes = plt.subplots(2, 1)
df[(df.index>'2018-01-01')&(df.index<'2019-01-01')].plot(ax=axes[0,],figsize=(12,9),grid=1,title='Dow Jones Index')
actsignal[(actsignal.index>'2018-01-01')&(actsignal.index<'2019-01-01')].plot(ax=axes[1,],figsize=(12,9),title='signal')
plt.show()

signal = []
for i in range(len(sss)):
    if sss.loc[i,'buys'] > sss.loc[i,'sells']:
        signal.append(1)
    elif sss.loc[i,'buys'] == sss.loc[i,'sells']:
        signal.append(3)
    elif sss.loc[i,'buys'] < sss.loc[i,'sells']:
        signal.append(0)
    else:
        signal.append(5)
sss['signal'] = signal
sss['shiftaction'] = sss['action'].shift(1)

precisiondf = sss.iloc[20:,-2:].reset_index(drop = True)
a = []
correct = 0
false = 0
equal = 0
for i in range(len(precisiondf)):
    if precisiondf.loc[i,'signal'] == 1:
        if precisiondf.loc[i,'shiftaction'] == 1:
            a.append('Correct')
        else:
            a.append('False')
    if precisiondf.loc[i,'signal'] == 0:
        if precisiondf.loc[i,'shiftaction'] == 0:
            a.append('Correct')
        else:
            a.append('False')
    if precisiondf.loc[i,'signal'] == 3:
        a.append('Equal')
precisiondf['compare'] = a
compareprecision = DataFrame(columns=['Correct', 'False','Equal'],index=['train','valid','test'])
aa = precisiondf['compare'][:math.ceil(len(precisiondf)*0.6)].value_counts()
bb = precisiondf['compare'][math.ceil(len(precisiondf)*0.6):math.ceil(len(precisiondf)*0.8)].value_counts()
cc = precisiondf['compare'][math.ceil(len(precisiondf)*0.8):].value_counts()
compareprecision['Correct'] = [aa['Correct'],bb['Correct'],cc['Correct']]
compareprecision['False'] = [aa['False'],bb['False'],cc['False']]
compareprecision['Equal'] = [aa['Equal'],bb['Equal'],cc['Equal']]
compareprecision['Precision'] = (compareprecision['Correct'] + compareprecision['Equal'])/(compareprecision['Correct'] + compareprecision['False'] + compareprecision['Equal'])   

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

#績效比較表
comparereturn = DataFrame(columns=['Dow Jones Index', 'DQN model return'],index=['train','valid','test'])
comparereturn.loc['train','DQN model return'] = totalreturnlist[math.ceil(len(totalreturnlist)*0.6)]
comparereturn.loc['valid','DQN model return'] = totalreturnlist[math.ceil(len(totalreturnlist)*0.8)] - totalreturnlist[math.ceil(len(totalreturnlist)*0.6)]
comparereturn.loc['test','DQN model return'] = totalreturnlist[-1] - totalreturnlist[math.ceil(len(totalreturnlist)*0.8)]
comparereturn.loc['train','Dow Jones Index'] = dataclose[math.ceil(len(dataclose)*0.6)] - dataclose[0]
comparereturn.loc['valid','Dow Jones Index'] = dataclose[math.ceil(len(dataclose)*0.8)] - dataclose[math.ceil(len(dataclose)*0.6)]
comparereturn.loc['test','Dow Jones Index'] = dataclose[-1] - dataclose[math.ceil(len(dataclose)*0.8)]
comparereturn['DQN Model Profit'] = comparereturn['DQN model return']*20
summary = pd.concat([compareprecision,comparereturn],axis = 1)


# In[]
sss1 = sss.iloc[20:,:]
sss1 = sss1.set_index('date')  
sss1['shiftaction'] = np.where(sss1['shiftaction'] == 1,-1,1)
sss1['return'] = sss1['sells']-sss1['buys'] 
sss1['totalreturn'] = sss1['shiftaction']*sss1['return']
sss1['totalreturn_long'] = np.where(sss1['shiftaction'] == 1,sss1['shiftaction']*sss1['return'],0)
sss1['totalreturn_short'] = np.where(sss1['shiftaction'] == -1,sss1['shiftaction']*sss1['return'],0)

yearlist = [i for i in range(2007,2021)]
years = []
long_times_list,short_times_list,average_long,average_short,years_total_profit = [],[],[],[],[]
for i in range(2007,2021):
    years.append(sss1[(str(i)<sss1.index)&(sss1.index<str(i+1))])
for i in range(len(years)):
    years_total_profit.append(round(sum(years[i]['totalreturn']),2))
    long_times = years[i]['shiftaction'].value_counts()[1]
    long_times_list.append(long_times)
    short_times = years[i]['shiftaction'].value_counts()[-1]
    short_times_list.append(short_times)
    average_long.append(sum(years[i]['totalreturn_long'])/long_times)
    average_short.append(sum(years[i]['totalreturn_short'])/short_times)
    
year_stat = DataFrame({'年度':yearlist,'做多次數':long_times_list,'做空次數':short_times_list,'做多平均報酬':average_long,
                '做空平均報酬':average_short,'年度報酬':years_total_profit})
year_stat = year_stat.set_index('年度')


# In[]
sss2 = sss.iloc[20:,:]
sss2 = sss2.set_index('date')
sss2['correct'] = np.where(sss2['signal'] == sss2['shiftaction'],1,0)
sss2['false'] = np.where(sss2['signal'] + sss2['shiftaction'] == 1,1,0)
sss2['equal'] = np.where(sss2['signal'] == 3,1,0)
sss2['totalreturn'] = sss1['totalreturn']
sss2['correctreturn'] = sss2['correct']*sss2['totalreturn']
sss2['falsereturn'] = sss2['false']*sss2['totalreturn']
sss2['equalreturn'] = sss2['equal']*sss2['totalreturn']

years = []
correctlist,falselist,equallist,precision,correctreturnlist,falsereturnlist,correctreturn_meanlist,falsereturn_meanlist = [],[],[],[],[],[],[],[]
for i in range(2007,2021):
    years.append(sss2[(str(i)<sss2.index)&(sss2.index<str(i+1))])
for i in range(len(years)):
    correct = sum(years[i]['correct'])
    false = sum(years[i]['false'])
    equal = sum(years[i]['equal'])
    correctreturn = sum(years[i]['correctreturn'])
    falsereturn = sum(years[i]['falsereturn'])
    correctlist.append(correct)
    falselist.append(false)
    equallist.append(equal)
    precision.append((correct+equal)/(correct+equal+false))
    correctreturnlist.append(correctreturn)
    falsereturnlist.append(falsereturn)
    correctreturn_meanlist.append(correctreturn/correct)
    falsereturn_meanlist.append(falsereturn/false)
year_precision = DataFrame(
    {'預測正確次數':correctlist,'預測錯誤次數':falselist,'收盤價格一樣次數':equallist,
     '準確率':precision,'預測正確的報酬':correctreturnlist,'預測正確的平均報酬':correctreturn_meanlist,
     '預測錯誤的損失':falsereturnlist,'預測錯誤的平均損失':falsereturn_meanlist},index = yearlist)


ohlc = getStockDataVec('DJI_2007', 'all')
dji = pd.DataFrame({'date':dateid,'open':ohlc[0],'high':ohlc[1],'low':ohlc[2],'close':ohlc[3],'signal':actionlist})
dji = dji.set_index('date')
dji.to_csv('dji_signal.csv')
