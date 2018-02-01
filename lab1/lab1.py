#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 08:17:14 2018

@author: zahra
"""

import nltk
from nltk.corpus import brown
from nltk.util import ngrams
from collections import Counter
import numpy as np
from sklearn.decomposition import TruncatedSVD
import math
import scipy as sp


n=5000
judged_pairs = [('gem','jewel',3.91),('midday','noon',3.91),('car','automobile',3.92),('cemetery','graveyard',3.88),('cushion','pillow',3.84),('boy','lad',3.82),('cock','rooster',3.68),('tool','implement',3.66),('forest','woodland',3.65),('coast','shore',3.60),('autograph','signature',3.59),('journey','voyage',3.58),('serf','slave',3.46),('grin','smile',3.46),('glass','tumbler',3.45),('cord','string',3.41),('hill','mound',3.45),('magician','wizard',3.21),('furnace','stove',3.11),('asylum','madhouse',3.64),('brother','monk',2.74),('food','fruit',2.69),('bird','cock',2.63),('bird','crane',2.63),('oracle','sage',2.61),('sage','wizard',2.46),('lad','brother',2.41),('crane','implement',2.37),('magician','oracle',1.82),('glass','jewel',1.78),('cemetery','mound',1.69),('journey','car',1.55),('hill','woodland',1.48),('crane','rooster',1.41),('furnace','implement',1.37),('coast','hill',1.26),('bird','woodland',1.24),('shore','voyage',1.22),('cemetery','woodland',1.18),('food','rooster',1.09),('forest','graveyard',1.00),('lad','wizard',0.99),('mound','shore',0.97),('automobile','cushion',0.97),('boy','sage',0.96),('monk','oracle',0.91),('shore','woodland',0.90),('grin','lad',0.88),('coast','forest',0.85),('asylum','cemetery',0.79),('monk','slave',0.57),('cushion','jewel',0.45),('boy','rooster',0.44),('glass','magician',0.44),('graveyard','madhouse',0.42),('asylum','monk',0.39),('asylum','fruit',0.19),('grin','implement',0.18),('mound','stove',0.14),('automobile','wizard',0.11),('autograph','shore',0.06),('fruit','furnace',0.05),('noon','string',0.04),('rooster','voyage',0.04),('chord','smile',0.02)]

def get_W():     
    #step 1
    words = brown.words() #punctuations? stop words?
    #step 2
    unigrams = ngrams(words, 1)
    unigrams = Counter(unigrams).most_common()
    W = unigrams[0:n]           
    for i in range(len(W)):
        W[i] = W[i][0][0]
    W = list(W)
    return words, W

def word_index(words, W):
    dic = {}
    for w in words:
        try:
            index = W.index(w)
        except (ValueError):
            index = -1
        dic[w] = index
    return dic

#step 3
def construct_M1(words, W, index):
    M1 = []
    for i in range(n):
        row = np.zeros(n)
        M1.append(row)
    M1 = np.array(M1)
    bigrams = ngrams(words, 2)
    bigrams = Counter(bigrams).most_common()
    #print(type(bigrams[0]), bigrams[0])
    for each in bigrams:
        index1 = index[each[0][1]]
        index2 = index[each[0][0]]
        count = each[1]
        if index1>=0 and index2>=0:
            M1[index1][index2] = count
    return M1

#step 4
def construct_M1_plus(M1):
    M1_plus = []
    for i in range(n):
        row = np.zeros(n)
        M1_plus.append(row)
    M1_plus = np.array(M1_plus)
    
    w_sum = M1.sum(axis=1)
    c_sum = M1.sum(axis=0)
    total = w_sum.sum()
    print (total)
    for i in range(n):#row-w
        for j in range(n):#column-c
            p_w_c = M1[i][j]/total
            p_w = w_sum[i]/total
            p_c = c_sum[j]/total
#            print (p_w_c, p_w, p_c, I)
            if p_w==0 or p_c==0:
                frac = 0
            else:
                frac = (p_w_c)/(p_w*p_c)
            PMI = np.log(frac)
            M1_plus[i][j] = max (PMI, 0.0)
    return M1_plus

def construct_M2(M, compo_num):
    
    print(M.shape)
    svd = TruncatedSVD(n_components=compo_num)
    svd.fit(M, (n, n))
#    print(svd.components_)
    M2 = svd.transform(M)
    print(M2.shape)
    
    return M2

def cosine_sim(vec1, vec2):
    cos = np.dot(vec1, vec2)
    x = math.sqrt(np.dot(vec1, vec1))
    y = math.sqrt(np.dot(vec2, vec2))
    if math.isnan(cos/(x*y)):
        print (x, y, cos)
    return  cos / (x * y)

def predict_sims(M, P, word_index):
    sims = []
    for pair in P:
        index1 = word_index[pair[0]]
        dist1 = M[index1]
        index2 = word_index[pair[1]]
        dist2 = M[index2]
        sim = cosine_sim(dist1, dist2)
        sims.append(sim)
    return sims

def main():
    words, W = get_W()
    w_index = word_index(words, W)
    M1 = construct_M1(words, W, w_index)
    #print (M1)
    M1_plus = construct_M1_plus(M1)
#    print (M1_plus)
    
    #step 5
    M2_10 = construct_M2(M1_plus, 10)
    print(M2_10.shape)
    M2_50 = construct_M2(M1_plus, 50)
    print(M2_50.shape)
    M2_100 = construct_M2(M1_plus, 100)
    print(M2_100.shape)
    
    #step 6
    P = []
    S = []
    for each in judged_pairs:
        if (each[0] in W) and (each[1] in W):
            P.append(each)
            S.append(each[2])
    print (P)
    #print (S)
    
    #step 7
    S_M1 = predict_sims(M1, P, w_index)
    print ('S_M1: ', S_M1)
    S_M1_plus = predict_sims(M1_plus, P, w_index)
    print ('S_M1_plus: ', S_M1_plus)
    S_M2_10 = predict_sims(M2_10, P, w_index)
    print ('S_M2_10: ', S_M2_10)
    S_M2_50 = predict_sims(M2_50, P, w_index)
    print ('S_M2_50: ', S_M2_50)
    S_M2_100 = predict_sims(M2_100, P, w_index)
    print ('S_M2_100: ', S_M2_100)
    
    #step 8
    pc_M1 = sp.stats.pearsonr(S, S_M1)
    print ('pearson-correlation coefficient for M1: ', pc_M1)
    pc_M1_plus = sp.stats.pearsonr(S, S_M1_plus)
    print ('pearson-correlation coefficient for M1_plus: ', pc_M1_plus)
    pc_M2_10 = sp.stats.pearsonr(S, S_M2_10)
    print ('pearson-correlation coefficient for M2_10: ', pc_M2_10)    
    pc_M2_50 = sp.stats.pearsonr(S, S_M2_50)
    print ('pearson-correlation coefficient for M2_50: ', pc_M2_50)
    pc_M2_100 = sp.stats.pearsonr(S, S_M2_100)
    print ('pearson-correlation coefficient for M2_100: ', pc_M2_100)
    
    pass

if __name__ == '__main__':
   main() 