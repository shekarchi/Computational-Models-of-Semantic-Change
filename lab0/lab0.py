#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 22:27:21 2018

@author: zahra
"""

import sys
import nltk
from nltk import word_tokenize
from string import punctuation
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import numpy as np

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def get_ngrams(text_file):
    tokenized_corpus = []
    with open(text_file, 'r') as corpus:
        string_corpus = corpus.read()
        string_corpus = strip_punctuation(string_corpus)
        string_corpus = string_corpus.replace('"', '')
        string_corpus = string_corpus.replace('’', '')
        string_corpus = string_corpus.replace('‘', '')
        string_corpus = string_corpus.replace('“', '')
        string_corpus = string_corpus.replace('”', '')
        tokenized_corpus = word_tokenize(string_corpus)
    unigrams = ngrams(tokenized_corpus, 1)
    unigrams = Counter(unigrams).most_common()
    bigrams = ngrams(tokenized_corpus, 2)
    bigrams = Counter(bigrams).most_common()
    trigrams = ngrams(tokenized_corpus, 3)
    trigrams = Counter(trigrams).most_common()
    #print (bigrams)
    return unigrams, bigrams, trigrams

def save_ngrams(unigrams, bigrams, trigrams, unigrams_file, bigrams_file, trigrams_file):
    with open(unigrams_file, 'w') as ugf:
        ugf.write('word,count\n')
        for item in unigrams:
            ugf.write (item[0][0] + ',' + str(item[1]) + '\n')
    ugf.close()
    with open(bigrams_file, 'w') as bgf:
        bgf.write('word,count\n')
        for item in bigrams:
            bgf.write (item[0][0] + ',' + item[0][1] + ',' + str(item[1]) + '\n')
    bgf.close()
    with open(trigrams_file, 'w') as tgf:
        tgf.write('word,count\n')
        for item in trigrams:
            tgf.write (item[0][0] + ',' + item[0][1] + ',' + item[0][2] + ',' + str(item[1]) + '\n')
    tgf.close()

def normalize(arr):
    total = 0.0
    res = []
    for i in arr:
        total += i
    for i in arr:
        res.append(i/total)
    return res

def get_sentence_from_onegram(unigrams, start, length):
    a = []
    p = []
    for item in unigrams:
        a.append(item[0][0])
        p.append(item[1])
    p = normalize(p)
    for n in range(length):
        next_word = np.random.choice(a,1,True,p)[0]
        start = start + next_word + ' '
    return start

def get_sentence_from_twogram(bigrams, start, length):
    bigrams_dict = dict()
    for item in bigrams:
        first_word = item[0][0]
        second_word = item[0][1]
        count = item[1]
        if first_word in bigrams_dict:
            bigrams_dict[first_word].append((second_word, count))
        else:
            bigrams_dict[first_word] = []
            bigrams_dict[first_word].append((second_word, count))
    
    sentence = start
    for i in range(length-1):
        a = []
        p = []
        for item in bigrams_dict[start]:
            a.append(item[0])
            p.append(item[1])
        p = normalize(p)
        next_word = np.random.choice(a,1,True,p)[0]
        sentence = sentence + ' ' + next_word
        start = next_word
    return sentence

def get_sentence_from_threegram(trigrams, start, after_start, length):
    trigrams_dict = dict()
    for item in trigrams:
        first_word = item[0][0]
        second_word = item[0][1]
        third_word = item[0][2]
        count = item[1]
        if first_word+'-'+second_word in trigrams_dict:
            trigrams_dict[first_word+'-'+second_word].append((third_word, count))
        else:
            trigrams_dict[first_word+'-'+second_word] = []
            trigrams_dict[first_word+'-'+second_word].append((third_word, count))
    sentence = start + ' ' + after_start
    for i in range(length-2):
        a = []
        p = []
        for item in trigrams_dict[start + '-' + after_start]:
            a.append(item[0])
            p.append(item[1])
        p = normalize(p)
        next_word = np.random.choice(a,1,True,p)[0]
        sentence = sentence + ' ' + next_word
        start = after_start
        after_start = next_word
    return sentence


def main():
    '''a text file should be provided for this function, text_file is the full name of it,
    names for saving 1-3-grams data should be provided in order.
    sample running: python lab0.py text_file.txt unigrams.csv bigrams.csv trigrams.csv
    '''
    
    text_file = sys.argv[1]
    unigrams, bigrams, trigrams = get_ngrams(text_file)
    unigrams_file = sys.argv[2]
    bigrams_file = sys.argv[3]
    trigrams_file = sys.argv[4]
    save_ngrams (unigrams, bigrams, trigrams, unigrams_file, bigrams_file, trigrams_file)
    
    print ('\nGenerating sentences by sampling from 1-gram counts')
    for i in range(6,11):
        sen = get_sentence_from_onegram(unigrams, '', i)
        print (sen)
    print ('\nGenerating sentences by sampling from 2-gram counts')
    for i in range(6,11):
        sen = get_sentence_from_twogram(bigrams, 'and', i)
        print (sen)
    print ('\nGenerating sentences by sampling from 3-gram counts')
    for i in range(6,11):
        sen = get_sentence_from_threegram(trigrams, 'and', 'the', i)
        print (sen)

if __name__ == "__main__":
    main()
    
    
    