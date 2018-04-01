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
 

def myplot(x,y, lx, ly):
    plt.scatter(x, y)
#    plt.title('Scatter plot pythonspot.com')
    plt.xlabel(lx)
    plt.ylabel(ly)
    plt.show()


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
    freqs = []
    freqs2 = []
    sense_num = []
    sense_num2 = []
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
        for yf in year_freq[1:-1]:
            yf = yf.split(',')
#            print(yf)
            y = yf[0]
            f = yf[1]
            freq += float(f)*100
            
        if data[2] > 0 :
            
#            with open('./load_ggl_ngram/'+w+'_NOUN-eng_2012-1500-2000-3-caseInsensitive.csv') as f:
#                content = f.readlines()
#            year_freq = [x.strip() for x in content]
#            freq = 0
#            for yf in year_freq[1:-1]:
#                yf = yf.split(',')
#                y = yf[0]
#                f = yf[1]
#                freq += float(f)*100
            
            
            
            if target_words_freq_nf_vf_totf[w][2]==433441.0 or target_words_freq_nf_vf_totf[w][2]==18188.0:
                continue
            d.append(data[2])
            len_.append(data[3])
#            freqs.append(np.log(target_words_freq_nf_vf_totf[w][2]))
#            freqs.append(target_words_freq_nf_vf_totf[w][0])
            freqs.append(freq)
            sense_num.append(data[4])
            
            len_2.append(data[3])
#            freqs2.append(np.log(target_words_freq_nf_vf_totf[w][2]))
#            freqs2.append(target_words_freq_nf_vf_totf[w][0])
            freqs2.append(freq)
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
            sense_num2.append(data[4])
            change.append(0)
            unchanged_nouns.append(w)
#    coeff = np.corrcoef(d,len_)
#    print(coeff)
    
    
    coeff = scipy.stats.pearsonr(d, len_)
    print('d and word length:', coeff)
    
    coeff = scipy.stats.pearsonr(d, freqs)
    print('d and word freq:', coeff)
    
    coeff = scipy.stats.pearsonr(d, sense_num)
    print('d and word sense num:', coeff)
    
    print(len(d), len(len_), len(freqs), len(sense_num), len(change), len(len_2), len(freqs2), len(sense_num2))
    
    coeff = scipy.stats.pearsonr(change, len_2)
    print('change and word length:', coeff)
    
    coeff = scipy.stats.pearsonr(change, freqs2)
    print('change and word freq:', coeff)
    
    coeff = scipy.stats.pearsonr(change, sense_num2)
    print('change and word sense num:', coeff)
    
    myplot(d,len_,'d','word length')
    myplot(d,freqs,'d','word freq')
    myplot(d,sense_num,'d','word sense num')    
    myplot(change,len_2,'change','word length')
    myplot(change,freqs2,'change','word freq')
    myplot(change,sense_num2,'change','word sense num')
    
    print(len(changed_nouns))
    print(changed_nouns)
    from_ = 1500
    o = open('changed_nouns_from_'+str(from_),"w")
    for n in changed_nouns:
        o.write(n+'\n')
    o.close()
#    for i in range(int(len(d))):
#        print(d[i], freqs[i], len_[i])
    
#    for i in range(int(len(change))):
#        print(change[i], freqs2[i], len_2[i])
    print(len(unchanged_nouns))
    o = open('unchanged_nouns_from_'+str(from_),"w")
    for n in unchanged_nouns:
        o.write(n+'\n')
    o.close()
    
    
    
    
    
    
    
    
    