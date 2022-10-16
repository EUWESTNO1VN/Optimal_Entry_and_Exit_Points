# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 19:47:34 2022

@author: wutia
"""
# In this file we write code about curve forecasting models
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pywt
from sklearn.cluster import AgglomerativeClustering
from skfda.preprocessing.dim_reduction.projection import FPCA
from skfda.representation.grid import FDataGrid
from pmdarima import AutoARIMA

# Absolute and squared forecast errors
def Mafe_t(real_data, pre_data):
    q = np.shape(real_data)[0]
    t = np.shape(real_data)[1]
    error = []
    for j in range(t):
        q_t = 0
        for i in range(q):
            q_t += np.abs(real_data[i,j]-pre_data[i,j])
        error.append(q_t/q)
    return error

def Msfe_t(real_data, pre_data):
    q = np.shape(real_data)[0]
    t = np.shape(real_data)[1]
    error=[]
    for j in range(t):
        q_t = 0
        for i in range(q):
            q_t += (real_data[i,j]-pre_data[i,j])**2
        error.append(q_t/q)
    return error

# FWK method
def distance_wave(data_1, data_2, dec_level, w):
    coef_1 = pywt.wavedec(data_1, w, level=dec_level)
    coef_2 = pywt.wavedec(data_2, w, level=dec_level)
    coef_1 = coef_1[1:]
    coef_2 = coef_2[1:]
    d = 0
    for i in range(len(coef_1)):
        c_1 = coef_1[i]
        c_2 = coef_2[i]
        d_i = 0
        for j in range(c_1.shape[0]):
            d_i += (c_1[j]-c_2[j])**2
        d_i = np.sqrt(d_i)
        d += np.sqrt(1/c_1.shape[0])*d_i
    return d

def K_h(x,h):
    value = (1/(h*np.sqrt(2*np.pi)))*np.exp(-0.5*((x/h)**2))
    return value

def FWK_method(train_set,w,dec_level, h_best=1):
    last_curve = train_set[-1,:]
    weight = np.zeros(train_set.shape[0]-1)
    sum_k = 0
    for k in range(train_set.shape[0]-1):
        d = distance_wave(train_set[k,:], last_curve, dec_level, w)
        weight[k] = K_h(d,h_best)
        sum_k += K_h(d,h_best)
    for j in range(weight.shape[0]):
        weight[j] = weight[j]/sum_k
    pre_day = np.zeros_like(last_curve)
    for j in range(pre_day.shape[0]):
        value_t = 0
        for k in range(len(weight)):
            value_t += weight[k]*train_set[k+1,j]
        pre_day[j] = value_t
    return pre_day    

def CFWK_method(train_set, w, dec_level, n_cluster, h_best=1):
    last_curve = train_set[-1,:]
    segment = train_set[:-1,:].copy()
    cluster_fwk =  AgglomerativeClustering(n_clusters=n_cluster).fit(segment)
    p = [0 for i in range(n_cluster)]
    k_sum = 0
    k_segment = np.zeros(segment.shape[0])
    for i in range(segment.shape[0]):
        d = distance_wave(segment[i,:], last_curve, dec_level, w)
        k = K_h(d, h_best)
        k_sum += k
        p[cluster_fwk.labels_[i]] += k
        k_segment[i] = k
    gm = p.index(max(p))
    n_gm = 0
    gm_sample = np.zeros_like(segment)
    plus_sample = np.zeros_like(segment)
    k_gm = np.zeros_like(k_segment)
    for i in range(segment.shape[0]):
        if cluster_fwk.labels_[i] == gm:
            gm_sample[n_gm,:] = segment[i,:]
            if i == segment.shape[0]-1:
                plus_sample[n_gm,:] = last_curve
            else:
                plus_sample[n_gm,:] = segment[i+1,:]
            k_gm[n_gm] = k_segment[i]
            n_gm += 1
    gm_sample = gm_sample[:n_gm,:]
    plus_sample = plus_sample[:n_gm,:]
    k_gm = k_gm[:n_gm]
    k_sum_gm = p[gm]
    pre_day = np.zeros_like(last_curve)
    for j in range(pre_day.shape[0]):
        value_t = 0
        for k in range(k_gm.shape[0]):
            value_t += (k_gm[k]/k_sum_gm)*plus_sample[k,j]
        pre_day[j] = value_t
    return pre_day

# ARMA method and VAR method
def get_principal_component(Fdata, time_day, time_interval, ratio):
    mu = []
    D = np.shape(Fdata)[0]
    T = np.shape(Fdata)[1]
    Fdata_c = Fdata.copy()
    for i in range(T):
        mu_t = np.mean(Fdata[:,i])
        mu.append(mu_t)
        for j in range(D):
            Fdata_c[j,i] = Fdata[j,i]- mu_t
    Fdata_c = np.array(Fdata_c,dtype='float')
    fd = pd.DataFrame(Fdata_c, columns=time_day)
    f_d = FDataGrid(fd, sample_points=time_interval)
    jug = True
    N=2
    while jug:
       fpca_n = FPCA(n_components=N)
       fpca_n.fit(f_d)
       if sum(fpca_n.explained_variance_ratio_)<ratio:
           N += 1
           continue
       else:
           jug = False
    phi = []
    for n in range(N):
        phi.append(fpca_n.components_[n])
    beta = fpca_n.transform(f_d)
    beta = pd.DataFrame(beta)
    return phi, mu, beta

def ts_forecast(mu, time_interval, phi, beta_predict):
    t_n = len(time_interval)
    p_n = len(phi)
    predict_day = len(beta_predict)
    predict_value = np.zeros((predict_day, t_n))
    for pre in range(predict_day):
        beta_day = beta_predict.iloc[pre].copy()
        for t in range(t_n):
            pre_value = 0
            for p in range(p_n):
                value = beta_day[p]*(phi[p](t))
                value = value.reshape((-1,))[0]
                pre_value += value
            predict_value[pre,t] = pre_value + mu[t]
    return predict_value

def ARMA_method(train_set,sample_times,ratio):
    time_interval = range(train_set.shape[1])
    phi_ts, mu_ts, beta_ts = get_principal_component(train_set, sample_times, 
                                                           time_interval, ratio)
    beta_arma = np.zeros(len(phi_ts))
    for i in range(len(phi_ts)):
        arma_model = AutoARIMA(start_p=1,start_q=1)
        beta_arma[i] = arma_model.fit_predict(beta_ts.iloc[:,i],n_periods=1)
    beta_arma = beta_arma.reshape((1,-1))
    beta_arma = pd.DataFrame(beta_arma)    
    pre_arma = ts_forecast(mu_ts, time_interval, phi_ts, beta_arma)
    return pre_arma

def VAR_method(train_set,sample_times,ratio,p,q):
    time_interval = range(train_set.shape[1])
    phi_var, mu_var, beta_var = get_principal_component(train_set, sample_times, 
                                                     time_interval, ratio)
    orgMod_var = sm.tsa.VARMAX(beta_var,order=(p,q),trend='ct',exog=None)
    fitMod_var = orgMod_var.fit(maxiter=100,disp=False)
    beta_pre_var = fitMod_var.predict(start=train_set.shape[0])
    pre_var = ts_forecast(mu_var, time_interval, phi_var, beta_pre_var)
    return pre_var


def curve_cluster(train_set, n_cluster=2):
    distance_matrix = np.zeros((train_set.shape[0],train_set.shape[1]-1))
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            distance_matrix[i,j] = train_set[i,j+1] - train_set[i,j]
    
    cluster =  AgglomerativeClustering(n_clusters=n_cluster).fit(distance_matrix)
    n_label = [0 for i in range(n_cluster)]
    for j in range(train_set.shape[0]):
        j_label = cluster.labels_[j]
        n_label[j_label] += 1
    extra_label = [i for i in range(len(n_label)) if n_label[i]==max(n_label)]
    
    last_label_in = False
    if len(extra_label) > 1:
        last_label = cluster.labels_[-1]
        for j in range(len(extra_label)):
            if last_label == extra_label[j]:
                last_label_in = True
                max_label = extra_label[j]
                break
    else:
        max_label = extra_label[0]
    if (not last_label_in) and (len(extra_label)>1):
        target_position = np.zeros(len(extra_label))
        for p,l in enumerate(extra_label):
            for h in range(train_set.shape[0]):
                if cluster.labels_[h] == l:
                    target_position[p] = h
        max_position = target_position.max()
        max_label = extra_label[max_position]
        
    same_label_sample = np.zeros_like(train_set)
    n_samelabel = 0
    for j in range(train_set.shape[0]):
        if cluster.labels_[j] == max_label:
            same_label_sample[n_samelabel,:] = train_set[j,:]
            n_samelabel += 1
    same_label_sample = same_label_sample[:n_samelabel,:]
    return same_label_sample

def arma_modify(train_set,sample_times,ratio):
    train = curve_cluster(train_set)
    pre_arma = ARMA_method(train, sample_times, ratio)
    return pre_arma

def var_modify(train_set, sample_times, ratio, p, q):
    train = curve_cluster(train_set)
    pre_var = VAR_method(train, sample_times, ratio, p, q)
    return pre_var