#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from tqdm import trange


# In[2]:


df = pd.read_csv('dji_signal.csv')
df['signal'] = np.where(df['signal'] == 0,1,-1)
df['signal_shift'] = df['signal'].shift(1)
dji = df['close']

# In[5]:


#BBands
def bbands(tsPrice,period,times):
    upBBand=pd.Series(0.0,index=tsPrice.index)
    midBBand=pd.Series(0.0,index=tsPrice.index)
    downBBand=pd.Series(0.0,index=tsPrice.index)
    sigma=pd.Series(0.0,index=tsPrice.index)
    for i in range(period,len(tsPrice)):
        midBBand[i]=round(np.nanmean(tsPrice[i-period:i]),4)
        sigma[i]=np.nanstd(tsPrice[i-period:i])
        upBBand[i]=round(midBBand[i]+times*sigma[i],4)
        downBBand[i]=round(midBBand[i]-times*sigma[i],4)
    BBands=pd.DataFrame({'upBBand':upBBand[(period):],'midBBand':midBBand[(period):],
                         'downBBand':downBBand[(period):],'sigma':sigma[(period):],                        'close':dji[(period):]})
    return(BBands)
djiBBands=bbands(dji,20,2)
djiBBands.head()


# In[6]:


plt.plot(dji,label="Close",color='k')
#plt.rcParams['font.sans-serif'] = ['simhei']
plt.plot(djiBBands.upBBand,
         label="upBBand",color='b',
         linestyle='dashed')
plt.plot(djiBBands.midBBand,
         label="midBBand",color='r',linestyle='-')
plt.plot(djiBBands.downBBand,
         label="downBBand",color='b',
         linestyle='dashed')
plt.xticks(rotation=45)
plt.title("djibband")
plt.legend()


# In[7]:



df_20 = pd.concat([df[20:],djiBBands],axis=1).iloc[:,:11]
df_20.head(15)


# In[9]:
df_20['sigma'].mean(),df_20['sigma'].median()

df_return = df_20['signal_shift']*(df_20['close']-df_20['open'])
filter = np.where((df_20['open']>df_20['upBBand'])|(df_20['open']<df_20['downBBand']),0,1)
df_return_filter = df_return*filter
df_return_filter[2138:2845].sum()
df_return.sum()

# In[]
result = {'profit':[],'day':[],'times':[]}
for i in trange(5,41):
    for j in range(3,100):
        djiBBands=bbands(dji,i,0.05*j)
        df_20 = pd.concat([df[i:],djiBBands],axis=1).iloc[:,:11]
        df_return = df_20['signal_shift']*(df_20['close']-df_20['open'])
        filter = np.where((df_20['open']>df_20['upBBand'])|(df_20['open']<df_20['downBBand']),0,1)
        df_return_filter = df_return*filter
        sum = df_return_filter[2138:2845].sum()
        result['profit'] .append(sum)
        result['day'] .append(i)
        result['times'] .append(j*0.05)
result_table = pd.DataFrame(result)
#40,3.65(all)
# In[60]:
df_20 = pd.concat([df[20:],bbands(dji,20,2)],axis=1).iloc[:,:10]
def test(period,times):
    df_20 = pd.concat([df[period:],bbands(dji,period,times)],axis=1).iloc[:,:10]

    #初始設定
    alert = 0
    observation = 0
    position = 0
    count = 0
    revenue = 0

    form = []
    x_long = []
    y_long = []
    x_short = []
    y_short = []

    for i in range(period+1,period+len(df_20)):    
        openlongpositionpoint = 0
        openshortpositionpoint = 0
        openlongpositionpoint1 = 0
        openshortpositionpoint1 = 0
        offsetpoint = 0
        openposition = 0 #記錄開倉點位
        offsetreason = 0 #平倉原因
        
        if alert == 3:
            position = 0
            openlongpositionpoint = 0
            openshortpositionpoint = 0
            offsetpoint = 0

        #開盤建立部位
        if df_20.loc[i,'downBBand'] <= df_20.loc[i,'open'] <= df_20.loc[i,'upBBand']:
        
            if alert == 2:
                if df_20.loc[i-1,'signal'] == 0:
                    position += 1
                    openlongpositionpoint = -df_20.loc[i,'open'] #建倉價格
                    openposition = df_20.loc[i,'open']
                    offsetpoint = 0
                    alert = 0

                if df_20.loc[i-1,'signal'] == 1:
                    position -= 1
                    openshortpositionpoint = df_20.loc[i,'open'] #建倉價格
                    openposition = df_20.loc[i,'open']
                    offsetpoint = 0
                    alert = 0

            if alert == 1:
                if position == 0:
                    if df_20.loc[i-1,'signal'] == 0:
                        position += 1
                        openlongpositionpoint = -df_20.loc[i,'open'] #建倉價格
                        openposition = df_20.loc[i,'open']
                        offsetpoint = 0
                        alert = 0

                    if df_20.loc[i-1,'signal'] == 1:
                        position -= 1
                        openshortpositionpoint = df_20.loc[i,'open'] #建倉價格
                        openposition = df_20.loc[i,'open']
                        offsetpoint = 0
                        alert = 0

        if alert == 1:
            if position == -1: #開盤空單平倉
                position = 0
                offsetpoint = -df_20.loc[i,'open'] #平倉價格
                alert = 2 #隔天要建倉
                offsetreason = 1

            if position == 1: #開盤多單平倉
                position = 0
                offsetpoint = df_20.loc[i,'open'] #平倉價格
                alert = 2 #隔天要建倉
                offsetreason = 1
        if alert == 5:
            if position == -1: #開盤空單平倉
                position = 0
                offsetpoint = -df_20.loc[i,'open'] #平倉價格
                offsetreason = 2
            if position == 1: #開盤多單平倉
                position = 0
                offsetpoint = df_20.loc[i,'open'] #平倉價格
                offsetreason = 2
            if df_20.loc[i-1,'signal'] == 0:
                position += 1
                openlongpositionpoint1 = -df_20.loc[i,'open'] #建倉價格
                openposition = df_20.loc[i,'open']
                alert = 0
            if df_20.loc[i-1,'signal'] == 1:
                position -= 1
                openshortpositionpoint1 = df_20.loc[i,'open'] #建倉價格
                openposition = df_20.loc[i,'open']
                alert = 0

        #盤中
        if alert == 0:    
            if position == 1: #持多單
                if df_20.loc[i,'low'] <= df_20.loc[i,'downBBand']:  #觸及下界
                    position -= 1 #平倉在下界
                    offsetpoint = df_20.loc[i,'downBBand']  #平倉價格(停損)
                    alert = 3  #假如盤中的position為0，設alert=3，等待下一次signal出現
                    offsetreason = 3

                if df_20.loc[i,'high'] >= df_20.loc[i,'upBBand'] and position == 1:  #觸及上界
                    position -= 1 #平倉在上界
                    offsetpoint = df_20.loc[i,'upBBand']  #平倉在上界(停利)
                    alert = 3  #假如盤中的position為0，設alert=3，等待下一次signal出現
                    offsetreason = 3


            if position == -1: #持空單
                if df_20.loc[i,'high'] >= df_20.loc[i,'upBBand']:  #觸及上界
                    position += 1 #平倉在上界
                    offsetpoint = -df_20.loc[i,'upBBand']  #平倉在上界(停損)
                    alert = 3  #假如盤中的position為0，設alert=3，等待下一次signal出現
                    offsetreason = 4
                    
                if df_20.loc[i,'low'] <= df_20.loc[i,'downBBand'] and position == -1:  #觸及下界
                    position += 1 #平倉在下界
                    offsetpoint = -df_20.loc[i,'downBBand']  #平倉價格(停利)
                    alert = 3  #假如盤中的position為0，設alert=3，等待下一次signal出現
                    offsetreason = 4

                


        #收盤
        try:
            if df_20.loc[i,'signal'] == df_20.loc[i-1,'signal'] and observation == 1: 
                alert = 5
                observation = 0
            if df_20.loc[i,'signal'] != df_20.loc[i-1,'signal'] and df_20.loc[i,'signal'] != df_20.loc[i-2,'signal']: #假如方向跟昨天不一樣
                observation = 1
            if df_20.loc[i,'signal'] != df_20.loc[i-1,'signal'] and df_20.loc[i,'signal'] == df_20.loc[i-2,'signal']:
                observation = 0
            if df_20.loc[i,'signal'] != df_20.loc[i-1,'signal']:
                if position == 1 and openposition > df_20.loc[i,'close']:#假如持多單且虧損就明天開盤平掉
                    alert = 1
                    observation = 0
                if position == -1 and openposition < df_20.loc[i,'close']:#假如持空單且虧損就明天開盤平掉
                    alert = 1
                    observation = 0
        except:
            pass

        #計算收益
        revenue += openlongpositionpoint + openshortpositionpoint + offsetpoint
        revenue = round(revenue,4)
        
        #計算交易次數
        if openlongpositionpoint != 0:
            count += 1
        if openshortpositionpoint != 0:
            count += 1
        if offsetpoint != 0:
            count += 1
        
        openlongpositionpoint = openlongpositionpoint + openlongpositionpoint1
        openshortpositionpoint = openshortpositionpoint + openshortpositionpoint1
            
        #紀錄x,y
        if openlongpositionpoint != 0:
            x_long.append(i)
            y_long.append(abs(openlongpositionpoint))
        if openshortpositionpoint != 0:
            x_short.append(i)
            y_short.append(abs(openshortpositionpoint))
        if offsetpoint > 0:
            x_short.append(i)
            y_short.append(abs(offsetpoint))
        if offsetpoint < 0:
            x_long.append(i)
            y_long.append(abs(offsetpoint))

        #print([openlongpositionpoint,openshortpositionpoint,offsetpoint,position,revenue])
        form.append([openlongpositionpoint,openshortpositionpoint,offsetpoint,position,revenue,count,df_20.loc[i,'close'],offsetreason,alert])
    return form,x_long,y_long,x_short,y_short


# In[61]:

test(15,1.7)[0]


# In[16]:
t=[]
for j in trange(5,40):
    for i in range(5,40):
        if test(j,0.1*i)[0][-1][3]==0:
            tt = test(j,0.1*i)[0][-1]
            tt.extend([j,i,-1])
            t.append(tt)
        elif test(j,0.1*i)[0][-2][3]==0:
            tt = test(j,0.1*i)[0][-2]
            tt.extend([j,i,-2])
            t.append(tt)
        elif test(j,0.1*i)[0][-3][3]==0:
            tt = test(j,0.1*i)[0][-3]
            tt.extend([j,i,-3])
            t.append(tt)
        elif test(j,0.1*i)[0][-4][3]==0:
            tt = test(j,0.1*i)[0][-4]
            tt.extend([j,i,-4])
            t.append(tt)
        elif test(j,0.1*i)[0][-5][3]==0:
            tt = test(j,0.1*i)[0][-5]
            tt.extend([j,i,-5])
            t.append(tt)
        else:
            print('wrong')
ttt = pd.DataFrame(t,columns=[str(i) for i in range(12)])

# In[41]:


plt.plot(dji,label="Close",color='k')
#plt.rcParams['font.sans-serif'] = ['simhei']
plt.plot(djiBBands.upBBand,
         label="upBBand",color='b',
         linestyle='dashed')
plt.plot(djiBBands.midBBand,
         label="midBBand",color='r',linestyle='-')
plt.plot(djiBBands.downBBand,
         label="downBBand",color='b',
         linestyle='dashed')
plt.xticks(rotation=45)
plt.title("2019-2020djibband")
plt.legend()
plt.scatter(test(15,1.7)[1],test(15,1.7)[2], color='green', s=20, marker="^")
plt.scatter(test(15,1.7)[3],test(15,1.7)[4], color='green', s=20, marker="v")
plt.rcParams["figure.figsize"] = (10, 8)


# In[62]:


#計算今年的報酬
test_20_12 = test(15,1.7)[0][-216:]
test_20_12


# In[63]:


test_20_12[0].append(0)
for i in range(1,len(test_20_12)):
    if test_20_12[i][3]==0 and test_20_12[i-1][3]==0:
        test_20_12[i].append(0)
    if test_20_12[i][3]==1:
        if test_20_12[i][0] != 0:
            test_20_12[i].append(test_20_12[i][0])
        else:
            test_20_12[i].append(test_20_12[i-1][7])
    if test_20_12[i][3]==-1:
        if test_20_12[i][1] != 0:
            test_20_12[i].append(test_20_12[i][1])
        else:
            test_20_12[i].append(test_20_12[i-1][7])
    if test_20_12[i][3]==0 and test_20_12[i-1][3]!=0:
        test_20_12[i].append(test_20_12[i-1][7])


# In[64]:


#計算累積損益、單次損益、交易次數
revenue = 0
win_count = 0
lose_count = 0
profit = []
loss = []
total_revenue = []
for i in range(len(test_20_12)):
    today_revenue = 0
    if test_20_12[i][7]==0 and test_20_12[i][2]==0:
        today_revenue = 0
    if test_20_12[i][7]==0 and test_20_12[i][2]!=0:
        today_revenue = test_20_12[i][0]+test_20_12[i][1]+test_20_12[i][2]
    if test_20_12[i][7]>0 and test_20_12[i][2]==0:
        today_revenue = test_20_12[i][7]-test_20_12[i][6]
    if test_20_12[i][7]>0 and test_20_12[i][2]!=0:
        today_revenue = test_20_12[i][7]+test_20_12[i][2]
    if test_20_12[i][7]<0 and test_20_12[i][2]==0:
        today_revenue = test_20_12[i][7]+test_20_12[i][6]
    if test_20_12[i][7]<0 and test_20_12[i][2]!=0:
        today_revenue = test_20_12[i][7]+test_20_12[i][2]
    total_revenue.append(round(revenue+today_revenue,4))
    if test_20_12[i][7]==0 and test_20_12[i][2]!=0: 
        revenue += today_revenue
        if today_revenue >= 0:
            win_count += 1
            profit.append([round(today_revenue,4),win_count])
        if today_revenue < 0:
            lose_count += 1
            loss.append([round(today_revenue,4),lose_count])
    if test_20_12[i][7]>0 and test_20_12[i][2]!=0:
        revenue += today_revenue
        if today_revenue >= 0:
            win_count += 1
            profit.append([round(today_revenue,4),win_count])
        if today_revenue < 0:
            lose_count += 1
            loss.append([round(today_revenue,4),lose_count])
    if test_20_12[i][7]<0 and test_20_12[i][2]!=0:
        revenue += today_revenue
        if today_revenue >= 0:
            win_count += 1
            profit.append([round(today_revenue,4),win_count])
        if today_revenue < 0:
            lose_count += 1
            loss.append([round(today_revenue,4),lose_count])

#total_revenue[0] = 0
total_revenue


# In[65]:


profit,loss


# In[66]:


from pandas.core.frame import DataFrame
df_profit = DataFrame(profit)
df_loss = DataFrame(loss)
df_profit.columns = ['損益','交易次數']
df_loss.columns = ['損益','交易次數']
df_profit


# In[67]:


df_profit.mean(),df_loss.mean(),df_profit.sum(),df_loss.sum()


# In[68]:


x = [i for i in range(216)]
plt.plot(x,total_revenue)
plt.xticks(rotation = 45)
plt.title('2020revenue(15,1.7)')


# In[32]:


#計算去年的報酬
test_20_12 = test(15,1.7)[0][:-216]
test_20_12


# In[33]:


test_20_12[0].append(0)
for i in range(1,len(test_20_12)):
    if test_20_12[i][3]==0 and test_20_12[i-1][3]==0:
        test_20_12[i].append(0)
    if test_20_12[i][3]==1:
        if test_20_12[i][0] != 0:
            test_20_12[i].append(test_20_12[i][0])
        else:
            test_20_12[i].append(test_20_12[i-1][7])
    if test_20_12[i][3]==-1:
        if test_20_12[i][1] != 0:
            test_20_12[i].append(test_20_12[i][1])
        else:
            test_20_12[i].append(test_20_12[i-1][7])
    if test_20_12[i][3]==0 and test_20_12[i-1][3]!=0:
        test_20_12[i].append(test_20_12[i-1][7])


# In[34]:


#計算累積損益、單次損益、交易次數
revenue = 0
win_count = 0
lose_count = 0
profit = []
loss = []
total_revenue = []
for i in range(len(test_20_12)):
    today_revenue = 0
    if test_20_12[i][7]==0 and test_20_12[i][2]==0:
        today_revenue = 0
    if test_20_12[i][7]==0 and test_20_12[i][2]!=0:
        today_revenue = test_20_12[i][0]+test_20_12[i][1]+test_20_12[i][2]
    if test_20_12[i][7]>0 and test_20_12[i][2]==0:
        today_revenue = test_20_12[i][7]-test_20_12[i][6]
    if test_20_12[i][7]>0 and test_20_12[i][2]!=0:
        today_revenue = test_20_12[i][7]+test_20_12[i][2]
    if test_20_12[i][7]<0 and test_20_12[i][2]==0:
        today_revenue = test_20_12[i][7]+test_20_12[i][6]
    if test_20_12[i][7]<0 and test_20_12[i][2]!=0:
        today_revenue = test_20_12[i][7]+test_20_12[i][2]
    total_revenue.append(round(revenue+today_revenue,4))
    if test_20_12[i][7]==0 and test_20_12[i][2]!=0: 
        revenue += today_revenue
        if today_revenue >= 0:
            win_count += 1
            profit.append([round(today_revenue,4),win_count])
        if today_revenue < 0:
            lose_count += 1
            loss.append([round(today_revenue,4),lose_count])
    if test_20_12[i][7]>0 and test_20_12[i][2]!=0:
        revenue += today_revenue
        if today_revenue >= 0:
            win_count += 1
            profit.append([round(today_revenue,4),win_count])
        if today_revenue < 0:
            lose_count += 1
            loss.append([round(today_revenue,4),lose_count])
    if test_20_12[i][7]<0 and test_20_12[i][2]!=0:
        revenue += today_revenue
        if today_revenue >= 0:
            win_count += 1
            profit.append([round(today_revenue,4),win_count])
        if today_revenue < 0:
            lose_count += 1
            loss.append([round(today_revenue,4),lose_count])
total_revenue


# In[35]:


profit,loss


# In[36]:


from pandas.core.frame import DataFrame
df_profit = DataFrame(profit)
df_loss = DataFrame(loss)
df_profit.columns = ['損益','交易次數']
df_loss.columns = ['損益','交易次數']
df_profit


# In[37]:


df_profit.mean(),df_loss.mean(),df_profit.sum(),df_loss.sum()


# In[38]:


x = [i for i in range(238)]
plt.plot(x,total_revenue)
plt.xticks(rotation = 45)
plt.title('2019revenue(15,1.7)')
plt.rcParams["figure.figsize"] = (10, 8)


# In[ ]:





# In[ ]:


