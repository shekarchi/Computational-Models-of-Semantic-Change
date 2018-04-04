#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 18:00:14 2018

@author: zahra
"""

import json
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.preprocessing import scale

def myplot(x,y, lx, ly):
    plt.scatter(x, y)
#    plt.title('Scatter plot pythonspot.com')
    plt.xlabel(lx)
    plt.ylabel(ly)
    plt.show()

def my_mean(arr):
    return sum(arr)/len(arr)

def my_var(arr):
    m = my_mean(arr)
    variance = sum([(ai - m)**2 for ai in arr]) / len(arr)
    return variance

def my_normalize(X):
    X = X.T
    for i in range(len(X)):
        X[i] = (X[i]-my_mean(X[i]))/np.sqrt(my_var(X[i]))
#    print(X.shape)
    return X.T

if __name__ == '__main__':
    fp = open('target_words_n_v_d_len', 'r')
    target_words_nt_vt_d_len = json.load(fp)
    print(len(target_words_nt_vt_d_len))
    fp = open('target_words_freq_n_v_tot', 'r')
    target_words_freq_nf_vf_totf = json.load(fp)
    
    print(len(target_words_freq_nf_vf_totf), len(target_words_nt_vt_d_len))
    print('for the words that have desired change, the pearson coefficients:')
    d = []
    len_ = []
    len_2 = []
    len_u = []
    freqs = []
    freqs2 = []
    freqs_u = []
    rec_freqs = []
    rec_freqs2 = []
    rec_freqs_u = []
    sense_num = []
    sense_num2 = []
    sense_num_u = []
    change = []
    counter = 0
    changed_nouns = []
    unchanged_nouns = []
    for w, data in target_words_nt_vt_d_len.items():
#        print(w, data)
        with open('./load_ggl_ngram/'+w+'_NOUN-eng_2012-1500-2000-3-caseInsensitive.csv') as f:
            content = f.readlines()
        year_freq = [x.strip() for x in content]
        freq = 0
#        print(year_freq[0])
        if len(year_freq[0])<5:
            print(w)
            continue
        
        verb_year = max(int(data[1])-1,1500)
#        print(len(year_freq), min(verb_year-1499,501))
        rec_freq = float(year_freq[min(verb_year-1499,501)].split(',')[1])#*1000
#        print(verb_year,verb_year-1499,rec_freq)
        
        for yf in year_freq[1:-1]:
            yf = yf.split(',')
#            print(yf)
            y = yf[0]
            f = yf[1]
            freq += float(f)#*1000
            
        if data[2] > 0 :
            
            if target_words_freq_nf_vf_totf[w][2]==433441.0 or target_words_freq_nf_vf_totf[w][2]==18188.0:
                continue
            d.append(data[2])
            len_.append(data[3])
#            freqs.append(np.log(target_words_freq_nf_vf_totf[w][2]))
#            freqs.append(target_words_freq_nf_vf_totf[w][0])
            freqs.append(freq)
            rec_freqs.append(rec_freq)
            sense_num.append(data[4])
            
            len_2.append(data[3])
#            freqs2.append(np.log(target_words_freq_nf_vf_totf[w][2]))
#            freqs2.append(target_words_freq_nf_vf_totf[w][0])
            freqs2.append(freq)
            rec_freqs2.append(rec_freq)
            sense_num2.append(data[4])
            change.append(1)
            
            changed_nouns.append(w)
        else:
            if target_words_freq_nf_vf_totf[w][2]==433441.0 or target_words_freq_nf_vf_totf[w][2]==18188.0:
                continue
            len_2.append(data[3])
#            freqs2.append(np.log(target_words_freq_nf_vf_totf[w][2]))
#            freqs2.append(target_words_freq_nf_vf_totf[w][0])
            freqs2.append(freq)
            rec_freqs2.append(rec_freq)
            sense_num2.append(data[4])
            len_u.append(data[3])
            freqs_u.append(freq)
            rec_freqs_u.append(rec_freq)
            sense_num_u.append(data[4])
            change.append(0)
            unchanged_nouns.append(w)
#    coeff = np.corrcoef(d,len_)
#    print(coeff)
    
    print(len(d), len(len_), len(freqs), len(rec_freqs), len(sense_num), len(change), len(len_2), len(freqs2), len(sense_num2), len(rec_freqs2))
    print(len(len_u), len(freqs_u), len(rec_freqs_u), len(sense_num_u))
    
    coeff = scipy.stats.pearsonr(d, len_)
    print('d and word length:', coeff)
    
    coeff = scipy.stats.pearsonr(d, freqs)
    print('d and word freq:', coeff)
    
    coeff = scipy.stats.pearsonr(d, rec_freqs)
    print('d and word rec_freq:', coeff)
    
    coeff = scipy.stats.pearsonr(d, sense_num)
    print('d and word sense num:', coeff)

    
#    coeff = scipy.stats.pearsonr(change, len_2)
#    print('change and word length:', coeff)
#    
#    coeff = scipy.stats.pearsonr(change, freqs2)
#    print('change and word freq:', coeff)
#    
#    coeff = scipy.stats.pearsonr(change, rec_freqs2)
#    print('change and word rec_freq:', coeff)
#    
#    coeff = scipy.stats.pearsonr(change, sense_num2)
#    print('change and word sense num:', coeff)
    
    myplot(d,len_,'d','word length')
    myplot(d,np.multiply(freqs,100),'d','word freq')
    myplot(d,np.multiply(100,rec_freqs),'d','word recent freq')
    myplot(d,sense_num,'d','word sense num') 
#    
    myplot(change,len_2,'change','word length')
    myplot(change,np.multiply(freqs2,100),'change','word freq')
    myplot(change,np.multiply(100,rec_freqs2),'change','word rec freq')
    myplot(change,sense_num2,'change','word sense num')
    
    print(len(changed_nouns))
#    print(changed_nouns)
    from_ = 1500
    o = open('changed_nouns_from_'+str(from_),"w")
    for n in changed_nouns:
        o.write(n+'\n')
    o.close()
#    for i in range(int(len(d))):
#        print(d[i], freqs[i], len_[i])
    
#    for i in range(int(len(change))):
#        print(change[i], freqs2[i], len_2[i])
#    print(len(unchanged_nouns))
#    o = open('unchanged_nouns_from_'+str(from_),"w")
#    for n in unchanged_nouns:
#        o.write(n+'\n')
#    o.close()
    
    X = []
    Y = []
    for i in range(int(len(d))) :
        Y.append(d[i])
#        x_i = (len_[i], freqs[i], sense_num[i])
#        x_i = (len_[i], freqs[i], rec_freqs[i], sense_num[i])
        x_i = (freqs[i], rec_freqs[i], sense_num[i])
        X.append(x_i)
    X = np.array(X)
    Y = np.array(Y)
    
    X = my_normalize(X)
    
    model = LinearRegression()
    model.fit(X, Y)
    params = model.coef_
    print('\nlin reg weights (wl, wf, wrf, ws) for d : ',params)
    
    X = []
    Y = []
    for i in range(int(len(change))) :
        Y.append(change[i])
#        x_i = (len_2[i], freqs2[i], sense_num2[i])
#        x_i = (len_2[i], freqs2[i], rec_freqs2[i], sense_num2[i])
        x_i = (len_2[i], freqs2[i],rec_freqs2[i])
        X.append(x_i)
    X = np.array(X)
    Y = np.array(Y)
    
    print(X.shape)
    X = my_normalize(X)
    
    model = LogisticRegression()
    model.fit(X, Y)
    params = model.coef_
    print('\nlog reg weights (wl, wf, wrf, ws) for ch: ',params)
    
    test_pred = model.predict(X)
    print (len(test_pred))
    accuracy = ((test_pred == Y).mean())
    print(accuracy)
    
    
    
    
#    data_to_plot = np.random.rand(100,5)
#    data_to_plot = np.random.rand(100,5)

#    fig = plt.figure(1, figsize=(9, 6))
#    fig, ax = plt.subplots(1,3, figsize=(9,4))
#    positions = np.arange(2) + 1
##    ax = fig.add_subplot(111)    
#    bp = ax[0].boxplot(X.T[0], showmeans=True)
#    bp = ax[1].boxplot(X.T[1], showmeans=True)
#    bp = ax[2].boxplot(X.T[2], showmeans=True)
#    plt.show()
    
    print('mean and std for word-len for changed and unchanged words:')
    print('mean: ',np.mean(len_), np.mean(len_u))
    print('std: ',np.std(len_),np.std(len_u))
    print('mean and std for word-accu-freq for changed and unchanged words:')
    print('mean: ',np.mean(freqs), np.mean(freqs_u))
    print('std: ',np.std(freqs),np.std(freqs_u))
    print('mean and std for word-recent_freq for changed and unchanged words:')
    print('mean: ',np.mean(rec_freqs), np.mean(rec_freqs_u))
    print('std: ',np.std(rec_freqs),np.std(rec_freqs_u))
    print('mean and std for word-sense_num for changed and unchanged words:')
    print('mean: ',np.mean(sense_num), np.mean(sense_num_u))
    print('std: ',np.std(sense_num),np.std(sense_num_u))
    
    
    print(len(len_), len(len_u), np.mean(len_), np.mean(len_u))
    tt = stats.ttest_ind(len_, len_u)
    print('t-test result for word len:', tt)
    
    tt = stats.ttest_ind(freqs, freqs_u)
    print('t-test result for word freq:', tt)
    
    tt = stats.ttest_ind(rec_freqs, rec_freqs_u)
    print('t-test result for word recent freq:', tt)
    
    tt = stats.ttest_ind(sense_num, sense_num_u)
    print('t-test result for word sense_num:', tt)
    
    