# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:13:08 2022

@author: wutia
"""
import pandas as pd
import numpy as np
import math
from scipy import signal

def extrem_strategy(data, long_only=False):
    max_point = signal.argrelextrema(data, np.greater)[0]
    max_point = list(max_point)
    min_point = signal.argrelextrema(data, np.less)[0]
    min_point = list(min_point)
    max_point.extend(min_point)
    extrem_point = max_point.copy()
    extrem_point.sort()
    e_point = []
    
    if len(max_point) == 0:
        return [(0,0)]
    if len(max_point) == 1:
        if len(min_point) == 0:
            if long_only:
                return [(max_point[0],0)]
            else:
                return [(max_point[0],0),(max_point[0],len(data)-1)]
        else:
            if long_only:
                return [(len(data)-1, max_point[0])]
            else:
                return [(0,max_point[0]),(len(data)-1,max_point[0])]
    
    if max_point[0] < min_point[0]:
        e_point.append(max_point[0])
        #left_label = max_point[0]
        for point in min_point:
            min_index = extrem_point.index(point)
            if min_index == len(extrem_point)-1:
                e_point.append(point)
            else:
                #left_point = extrem_point[min_index-1]
                right_point = extrem_point[min_index+1]
                if right_point in max_point:
                    e_point.append(point)
                    e_point.append(right_point)
                elif right_point not in max_point:
                    continue
    else:
        e_point.append(min_point[0])
        for point in max_point:
            max_index = extrem_point.index(point)
            if max_index == len(extrem_point)-1:
                e_point.append(point)
            else:
                right_point = extrem_point[max_index+1]
                if right_point in min_point:
                    e_point.append(point)
                    e_point.append(right_point)
                elif right_point not in min_point:
                    continue
    if max_point[0] < min_point[0]:
        strategy = [(e_point[0],0)]
        if long_only:
            for i in range(1,len(e_point),2):
                if i == len(e_point)-1:
                    strategy.append((len(data)-1,e_point[i]))
                    return strategy
                else:
                    strategy.append((e_point[i+1], e_point[i]))
        else:
            for i in range(len(e_point)):
                if i == len(e_point)-1:
                    if i%2 == 0:
                        strategy.append((e_point[i],len(data)-1))
                    else:
                        strategy.append((len(data)-1,e_point[i]))
                    return strategy
                else:
                    if i%2 == 0:
                        strategy.append((e_point[i],e_point[i+1]))
                    else:
                        strategy.append((e_point[i+1],e_point[i]))
    else:
        strategy = []
        if long_only:
            for i in range(len(e_point),2):
                if i == len(e_point)-1:
                    strategy.append((len(data)-1,e_point[i]))
                    return strategy
                else:
                    strategy.append((e_point[i+1],e_point[i]))
        else:
            strategy.append((0,e_point[0]))
            for i in range(len(e_point)):
                if i == len(e_point)-1:
                    if i%2 == 0:
                        strategy.append((len(data)-1,e_point[i]))
                    else:
                        strategy.append((e_point[i],len(data)-1))
                    return strategy
                else:
                    if i%2 == 0:
                        strategy.append((e_point[i+1],e_point[i]))
                    else:
                        strategy.append((e_point[i],e_point[i+1]))
    return strategy

def strategy_return(financial_data,strategy):
    r_sum = 1
    for k in range(len(strategy)):
        s = strategy[k]
        short_point = s[0]
        long_point = s[1]
        r = (financial_data[short_point] - financial_data[long_point])/financial_data[long_point]
        r_sum *= 1+r
    return r_sum-1

def pwp(ret):
    # P/L ratio
    ret_sub_1 = ret[ret>0].copy()
    ret_sub_2 = ret[ret<0].copy()
    pnl = np.abs(ret_sub_1.mean()/ret_sub_2.mean())
    # winning rate
    winrate = len(ret_sub_1)/len(ret)
    # Expectation Index
    expind = (1+np.abs(pnl))*winrate - 1
    return [pnl, winrate, expind]

def pwp_apply(ret, number):
    number = int(number)
    l = pwp(ret)
    return l[number]

def maxDDD(cumret,ddtype=0):
    N = len(cumret)
    highmark = np.zeros((N,1))
    drawdown = np.zeros((N,1))
    duration = np.zeros((N,1))
    for i in range(1,N):
        highmark[i] = max(highmark[i-1], cumret[i])
        if ddtype==0:
            drawdown[i] = 1-(1+cumret[i])/(1+highmark[i])
        elif ddtype > 0:
            drawdown[i] = (1+highmark[i])/(1+cumret[i]) - 1
        if drawdown[i] != 0:
            duration[i] = duration[i-1] + 1
    to = np.argmax(drawdown)
    drawdown_sub = drawdown[:to+1]
    z = np.where(drawdown_sub==0)
    z = z[0]
    fro = z[0].max()
    return [max(drawdown),max(duration),drawdown,duration,fro,to]

def maxDDD_apply(cret, number):
    number = int(number)
    l = maxDDD(cret)
    return l[number]
    
    
def matrix_stats(out, datatype):
    mu = out.mean(axis=0)
    mu = mu*datatype
    sg = out.std(axis=0)
    sg = sg*np.sqrt(datatype)
    sr = mu/sg
    maxr = out.max(axis=0)
    minr = out.min(axis=0)
    out_cu = 1+out
    cret = out_cu.cumprod(axis=0) - 1
    cumr = cret.iloc[-1]
    maxDD = cret.apply(maxDDD_apply,args=(0,))
    maxDD = maxDD.iloc[0]
    maxDu = cret.apply(maxDDD_apply,args=(1,))
    maxDu = maxDu.iloc[0]
    pnl = out.apply(pwp_apply,args=(0,))
    wnr = out.apply(pwp_apply,args=(1,))
    exi = out.apply(pwp_apply,args=(2,))
    stat = pd.DataFrame(columns=out.columns)
    stat = stat.append([mu, sg, sr, maxr, minr, cumr, maxDD, maxDu, pnl, wnr, exi],ignore_index=True)
    rowname = ["Average","Volatility","Sharpe","Max","Min","Cumulative",
               "Drawdown","Duration","Profit/Loss","Win Rate","Expectation"]
    stat.index = rowname
    return stat

def mean_point_strategy(financial_data,long_only=False):
    d_day = financial_data.shape[0]
    max_index = []
    min_index = []
    strategy = []
    for d in range(d_day):
        price_d = pd.DataFrame(financial_data[d,:],columns=['price'])
        max_t = price_d['price'].idxmax(axis=0)
        min_t = price_d['price'].idxmin(axis=0)
        max_index.append(max_t)
        min_index.append(min_t)
    max_mean = round(np.mean(max_index))
    min_mean = round(np.mean(min_index))
    if not long_only:
        strategy.append((max_mean, min_mean))
    else:
        if max_mean >= min_mean:
            strategy.append((max_mean, min_mean))
    return strategy

def benchmark_stat(data_set,start_date):
    test_set = data_set[data_set.index>=start_date]
    test_index = test_set.index
    train_set = data_set[data_set.index<start_date]
    test_set = np.array(test_set,dtype='float')
    d_test = test_set.shape[0]
    train_set = np.array(train_set, dtype='float')
    d_train = train_set.shape[0]
    data_set = np.array(data_set, dtype='float')
    
    b1_dayreturn = []
    for i in range(d_test):
        r = (test_set[i,-1] - test_set[i,0])/test_set[i,0]
        b1_dayreturn.append(r)
    
    b2_dayreturn = []
    b2_dayreturn.append((test_set[0,-1] - test_set[0,0])/test_set[0,0])
    for i in range(1,d_test):
        b2_dayreturn.append((test_set[i,-1]-test_set[i-1,-1])/test_set[i-1,-1])
    
    b3_dayreturn_l = []
    b3_dayreturn_ls = []

    for d in range(d_test):
        day_price = data_set[d_train+d,:]
        pre_price = data_set[d_train+d-1,:]
        strategy_l = extrem_strategy(pre_price,long_only=True)
        strategy_ls = extrem_strategy(pre_price,long_only=False)
        return_l =strategy_return(day_price, strategy_l)
        return_ls = strategy_return(day_price, strategy_ls)
        b3_dayreturn_l.append(return_l)
        b3_dayreturn_ls.append(return_ls)
    
    k = np.linspace(2,30,29)
    b4_dayreturn_l = np.zeros((d_test,len(k)))
    b4_dayreturn_ls = np.zeros((d_test,len(k)))
    for n in k:
        n = int(n)
        for d in range(d_test):
            past_data = data_set[d_train+d-n:d_train+d,:]
            day_price = data_set[d_train+d,:]
            
            strategy_l = mean_point_strategy(past_data,long_only=True)
            return_l = strategy_return(day_price,strategy_l)
            b4_dayreturn_l[d,n-2] = return_l
            
            strategy_ls = mean_point_strategy(past_data,long_only=False)
            return_ls = strategy_return(day_price,strategy_ls)
            b4_dayreturn_ls[d,n-2] = return_ls

    d_return_b4_l = pd.DataFrame(b4_dayreturn_l)
    d_return_b4_ls = pd.DataFrame(b4_dayreturn_ls)
            
    stat_b4_l = matrix_stats(d_return_b4_l, 252)
    stat_b4_ls = matrix_stats(d_return_b4_ls, 252)
    b4_l_best = stat_b4_l.loc['Cumulative'].idxmax()
    b4_ls_best = stat_b4_ls.loc['Cumulative'].idxmax() 
    
    benchmark_data = pd.DataFrame()
    benchmark_data['BENCHMARK1'] = b1_dayreturn
    benchmark_data.index = test_index
    benchmark_data['BENCHMARK2'] = b2_dayreturn
    benchmark_data['BENCHMARK3_l'] = b3_dayreturn_l
    benchmark_data['BENCHMARK3_ls'] = b3_dayreturn_ls
    benchmark_data['BENCHMARK4_l'] = list(d_return_b4_l[b4_l_best])
    benchmark_data['BENCHMARK4_ls'] = list(d_return_b4_ls[b4_ls_best])
    
    benchmark_stat = matrix_stats(benchmark_data, 252)
    
    return benchmark_data, benchmark_stat
    