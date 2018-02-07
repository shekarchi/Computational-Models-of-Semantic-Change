#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:33:17 2018

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

import sys
import gensim
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

judged_pairs = [('gem','jewel',3.91),('midday','noon',3.91),('car','automobile',3.92),('cemetery','graveyard',3.88),('cushion','pillow',3.84),('boy','lad',3.82),('cock','rooster',3.68),('tool','implement',3.66),('forest','woodland',3.65),('coast','shore',3.60),('autograph','signature',3.59),('journey','voyage',3.58),('serf','slave',3.46),('grin','smile',3.46),('glass','tumbler',3.45),('cord','string',3.41),('hill','mound',3.45),('magician','wizard',3.21),('furnace','stove',3.11),('asylum','madhouse',3.64),('brother','monk',2.74),('food','fruit',2.69),('bird','cock',2.63),('bird','crane',2.63),('oracle','sage',2.61),('sage','wizard',2.46),('lad','brother',2.41),('crane','implement',2.37),('magician','oracle',1.82),('glass','jewel',1.78),('cemetery','mound',1.69),('journey','car',1.55),('hill','woodland',1.48),('crane','rooster',1.41),('furnace','implement',1.37),('coast','hill',1.26),('bird','woodland',1.24),('shore','voyage',1.22),('cemetery','woodland',1.18),('food','rooster',1.09),('forest','graveyard',1.00),('lad','wizard',0.99),('mound','shore',0.97),('automobile','cushion',0.97),('boy','sage',0.96),('monk','oracle',0.91),('shore','woodland',0.90),('grin','lad',0.88),('coast','forest',0.85),('asylum','cemetery',0.79),('monk','slave',0.57),('cushion','jewel',0.45),('boy','rooster',0.44),('glass','magician',0.44),('graveyard','madhouse',0.42),('asylum','monk',0.39),('asylum','fruit',0.19),('grin','implement',0.18),('mound','stove',0.14),('automobile','wizard',0.11),('autograph','shore',0.06),('fruit','furnace',0.05),('noon','string',0.04),('rooster','voyage',0.04),('chord','smile',0.02)]

def get_W(n):
    words = brown.words() #punctuations? stop words?
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

def add_missed_words(W, Q, w_index, ind):
    new_words = set()
    for pair in Q:
        new_words.add(pair[0])
        new_words.add(pair[1])
    new_words = list(new_words)
    for w in new_words:
        W.append(w)
        w_index[w] = ind
        ind = ind + 1
    return W, w_index, ind

def construct_M1(words, W, index, n):
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
            M1[index1][index2] += count
            M1[index2][index1] += count
    return M1

def construct_M1_plus(M1, n):
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
            if p_w==0 or p_c==0:
                frac = 0
            else:
                frac = (p_w_c)/(p_w*p_c)
            PMI = np.log(frac)
            M1_plus[i][j] = max (PMI, 0.0)
    return M1_plus

def construct_M2(M, compo_num, n):

    print(M.shape)
    svd = TruncatedSVD(n_components=compo_num)
    svd.fit(M, (n, n))
    M2 = svd.transform(M)
    print(M2.shape)

    return M2

def cosine_sim(vec1, vec2):
    cos = np.dot(vec1, vec2)
    x = math.sqrt(np.dot(vec1, vec1))
    y = math.sqrt(np.dot(vec2, vec2))
    if math.isnan(cos/(x*y)):
        print (x, y, cos)
        return 0.0
    return  cos / (x * y)

def extract_word_embedding(words):
    from gensim.models.keyedvectors import KeyedVectors

    # this is how you load the model
    model = KeyedVectors.load_word2vec_format('/home/class_test/word2vec_pretrain_vec/GoogleNews-vectors-negative300.bin', binary=True)

    # to extract word vector
    vectors = {}
    for w in words:
        try:
            vectors[w] = model[w]
            #print(w, model[w])
        except (KeyError):
            print(w, 'has not been found')
            continue
    return vectors

def make_LSA_model(M, words, word_index):
    model = {}
    for w in words:
        index = word_index[w]
        vect = M[index]
#        print (type(vect), vect)
        model[w] = vect
    print('\n#LSA model has been made of length: ', len(model))
    return model


def find_intersected_data(LSA_model, word2vec_model, input_file):
    datafile = open(input_file, "r")
    line = datafile.readline()
    intersected_data = []
    while line:
        words = line.strip().split()
        if words[0] == ":":
            print (line)
            line = datafile.readline()
            continue
        try:
            lsa_vec0 = LSA_model[words[0]]
            lsa_vec1 = LSA_model[words[1]]
            lsa_vec2 = LSA_model[words[2]]
            lsa_vec3 = LSA_model[words[3]]
            w2vec0 = word2vec_model[words[0]]
            w2vec1 = word2vec_model[words[1]]
            w2vec2 = word2vec_model[words[2]]
            w2vec3 = word2vec_model[words[3]]
            intersected_data.append(line)
            line = datafile.readline()
        except (KeyError):
            line = datafile.readline()
            continue
    print('\n# Size of the interseceted data is: ', len(intersected_data))
    return intersected_data

def predict_closest_word(W1, W2, W3):
    p = []
    for i in range(len(W1)):
        res = W1[i]-W2[i]+W3[i]
        p.append(res)
    return np.array(p)

def find_closest_word(model, w):
    max_word = ''
    max_sim = -10
    for word, vector in model.items():
        sim = cosine_sim(vector, w)
        if sim > max_sim:
            max_word = word
            max_sim = sim
    return max_word, max_sim

def compute_accuracy_given_model(model, data):
    trues = 0
    total = 0
    for line in data:
        words = line.strip().split()
        if words[0] == ":":
            print (line)
            continue
        total += 1
        W1 = model[words[0]]
        W2 = model[words[1]]
        W3 = model[words[2]]
        Pred = predict_closest_word(W1, W2, W3)
        Pred, sim = find_closest_word(model, Pred)
        if Pred == words[3]:
            trues += 1
    return trues/total

def compute_accuracy_w2v(model, data):
    trues = 0
    total = 0
    for line in data:
        words = line.strip().split()
        if words[0] == ":":
            print (line)
            continue
        total += 1
        pred = model.most_similar(positive=[words[0], words[1]], negative=[words[2]])
        if pred[0][0] == words[3]:
            trues += 1
    return trues/total

def main():
    n=5000
    words, W = get_W(n)
    w_index = word_index(words, W)
    ind = len(W)

    #step 1
    P = []
    Q = []
    S = []
    for each in judged_pairs:
        if (each[0] in W) and (each[1] in W):
            P.append(each)
            S.append(each[2])
        else:
            Q.append(each)
    W, w_index, ind = add_missed_words(W, Q, w_index, ind)
    n = ind
    print(len(words), len(W), len(w_index), ind)

    P = []
    S = []
    for each in judged_pairs:
        if (each[0] in W) and (each[1] in W):
            P.append(each)
            S.append(each[2])

    print (len(P))

    M1 = construct_M1(words, W, w_index, n)
    M1_plus = construct_M1_plus(M1, n)
    M2_100 = construct_M2(M1_plus, 100, n)
    print(M2_100.shape)
    #step 4
    print (len(W), len(w_index))
    LSA_100 = make_LSA_model(M2_100, W, w_index)
    
    from gensim.models.keyedvectors import KeyedVectors
    w2v_model = KeyedVectors.load_word2vec_format('../../Data/GoogleNews-vectors-negative300.bin', binary=True)

    int_sem_data = find_intersected_data(LSA_100, w2v_model, './analogy_semantic_test_data')
    
    LSA_semantic_accuracy = compute_accuracy_given_model(LSA_100, int_sem_data)
    print('LSA_100 accuracy for semantic data: ', LSA_semantic_accuracy)#LSA_100 accuracy for semantic data:  0.008403361344537815
    
    w2v_semantic_accuracy = compute_accuracy_w2v(w2v_model, int_sem_data)
    print('word2vec accuracy for semantic data: ', w2v_semantic_accuracy) #0.0
    
    
    int_syn_data = find_intersected_data(LSA_100, w2v_model, '../../Data/analogy_syntactic_test_data')
    
    w2v_syntactic_accuracy = compute_accuracy_w2v(w2v_model, int_syn_data)
    print('word2vec accuracy for syntactic data: ', w2v_syntactic_accuracy)
    
    LSA_syntactic_accuracy = compute_accuracy_given_model(LSA_100, int_syn_data)
    print('LSA_100 accuracy for syntactic data: ', LSA_syntactic_accuracy)

    pass

if __name__ == '__main__':
   main()
