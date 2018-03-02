'''
for all words w in the list in the assignment:
    *** take top n (e.g. 20) similar words for w, for each decade
    find the rank-biased overlap (https://github.com/ragrawal/measures) between all decades
    find the correlation (and the p-value of the correlation) between time and rank-biased overlap
    analyze the results (plot them, do they make sense? etc.)
'''

import sys
import pandas as pd
import numpy as np
import gensim
from gensim.models.keyedvectors import KeyedVectors
import os.path
from pathlib import Path
from collections import defaultdict
import myRBO


#right now I just assume the list of the target words is available here:
#words = ['gay', 'mouse', 'zero', 'abondon', 'star', 'idea', 'man', 'girl'] #for simplicity now I assume that this is a List
decades = range(1850, 2050, 50) #for example

def load_target_words(filepath):
    words = []
    f = open(filepath)
    w = f.readline().strip()
    while w:
        words.append(w)
        w = f.readline().strip()
    print('the number of target words: ', len(words))
    words = set(words)
    words = list(words)
    print('the number of target words: ', len(words))
    return words

def top_n_sim_words (lang_model, word, n):
    top_n = lang_model.most_similar(positive=word, topn=n) #the result can be a list of (word, score)
    top_n_words = []
    for t in top_n:
        top_n_words.append(t[0])
    return top_n_words

def model_semantic_change_given_words(lang_model, words, times=decades):
    semantic_change_model = defaultdict(list) # it is a dictionary of {word, list[list_of_top_n_words]
    n = 20
    for time in times:
        lang_model_t = lang_model[time]
        for word in words:
            try:
                top_n_words = top_n_sim_words(lang_model_t, word, n) #the result can be a list of words
                semantic_change_model[word].append(top_n_words)
            except KeyError:
                continue
    return semantic_change_model

def analyze_and_save_change(word, top_n_sim, change_rates, handle):
    if len(top_n_sim) <= 2:
        return word, -100.0
    total_change = 0
    #print('== top ns and change rates size: ', len(top_n_sim), len(change_rates))
    for i in range(int(len(top_n_sim)-1)):
        handle.write('top 10 similar words @ time: ')
        handle.write(str(i)+' : ')
        for t in range(9):
            handle.write(top_n_sim[i][t]+', ')
        handle.write(top_n_sim[i][9]+ '\n')
        total_change -= change_rates[i]
    total_change = total_change/len(change_rates)
    n = int(len(top_n_sim)-1)
    handle.write('top 10 similar words @ time: ')
    handle.write(str(n)+' : ')
    for t in range(9):
        handle.write(top_n_sim[n][t]+', ')
    handle.write(top_n_sim[n][9]+ '\n')
    handle.write('{} # the average change is: {}\n\n'.format(word, total_change))
    return word, total_change


def load_lang_models_files(args, times):
    file_t1 = args[1]
    file_t2 = args[2]
    file_t3 = args[3]
    file_t4 = args[4]
    lang_model = {}

    my_file = Path(file_t1+".txt")
    if my_file.is_file():
        print('making the lang-model of time1 ...')
        model_t1 = KeyedVectors.load_word2vec_format(file_t1+".txt", binary=False)
        lang_model[times[0]] = model_t1
        print('making the lang-model of time2 ...')
        model_t2 = KeyedVectors.load_word2vec_format(file_t2+".txt", binary=False)
        lang_model[times[1]] = model_t2
        print('making the lang-model of time3 ...')
        model_t3 = KeyedVectors.load_word2vec_format(file_t3+".txt", binary=False)
        lang_model[times[2]] = model_t3
        print('making the lang-model of time4 ...')
        model_t4 = KeyedVectors.load_word2vec_format(file_t4+".txt", binary=False)
        lang_model[times[3]] = model_t4
        return lang_model


    print('making the lang-model of time1 ...')
    df_t1 = pd.read_pickle(file_t1, compression='infer')
    np.savetxt(file_t1+".txt", df_t1.reset_index().values, delimiter=" ", header="{} {}".format(len(df_t1), len(df_t1.columns)),comments="",fmt=["%s"] + ["%.18e"]*len(df_t1.columns))
    model_t1 = KeyedVectors.load_word2vec_format(file_t1+".txt", binary=False)
    lang_model[times[0]] = model_t1

    print('making the lang-model of time2 ...')
    df_t2 = pd.read_pickle(file_t2, compression='infer')
    np.savetxt(file_t2+".txt", df_t2.reset_index().values, delimiter=" ", header="{} {}".format(len(df_t2), len(df_t2.columns)),comments="",fmt=["%s"] +    ["%.18e"]*len(df_t2.columns))
    model_t2 = KeyedVectors.load_word2vec_format(file_t2+".txt", binary=False)
    lang_model[times[1]] = model_t2

    print('making the lang-model of time3 ...')
    df_t3 = pd.read_pickle(file_t3, compression='infer')
    np.savetxt(file_t3+".txt", df_t3.reset_index().values, delimiter=" ", header="{} {}".format(len(df_t3), len(df_t3.columns)),comments="",fmt=["%s"] +    ["%.18e"]*len(df_t3.columns))
    model_t3 = KeyedVectors.load_word2vec_format(file_t3+".txt", binary=False)
    lang_model[times[2]] = model_t3

    print('making the lang-model of time4 ...')
    df_t4 = pd.read_pickle(file_t4, compression='infer')
    np.savetxt(file_t4+".txt", df_t4.reset_index().values, delimiter=" ", header="{} {}".format(len(df_t4), len(df_t4.columns)),comments="",fmt=["%s"] +    ["%.18e"]*len(df_t4.columns))
    model_t4 = KeyedVectors.load_word2vec_format(file_t4+".txt", binary=False)
    lang_model[times[3]] = model_t4

    return lang_model


def main():
    #preparing the language model that is built already
    args = sys.argv
    if len(args) != 6:
        print('you should provide 5 filepath to lang models.')
        exit(-1)
    words = load_target_words(args[5])
    lang_model = load_lang_models_files(args, decades)
    print ('The language model has been loaded.')


    #lang_model = {} # I assume it is a dict of word2vec models; lang_model[time] is
                    # a word2vec - time is in range(1850, 2000, 50) for example.
    print('computing the semantic changes ...')
    semantic_change_model = model_semantic_change_given_words(lang_model, words, decades)
    print ('computing the semantic changes has been done.\n\nanalyzing changes ...')
    semantic_changes = {}
    output = open('semantic_changes.out', "w")
    word_change_mapping = []
    for word, top_ns in semantic_change_model.items():
        change_rates = [] # must be of size len(times)-1
        p = 0.9
        for i in range(int(len(top_ns)-1)):
            change = myRBO.score(top_ns[i], top_ns[i+1], p)
            #print('for word: <{}> at time {}, the change score is: {}'.format(word, i+1, change))
            change_rates.append(change)
        semantic_changes[word] = change_rates
        w, chng = analyze_and_save_change(word, top_ns, change_rates, output)
        if chng > -10:
            word_change_mapping.append((word, chng))

    word_change_mapping = sorted(word_change_mapping, key=lambda ch: ch[1])
    M = 200
    print('\n== The top M={} least changing words:'.format(M))
    least_changing = []
    for i in range(M):
        print (word_change_mapping[i])
        least_changing.append(word_change_mapping[i][0])
    print('\n== The top M={} most changing words:'.format(M))
    most_changing = []
    for i in range(1,M+1):
        print(word_change_mapping[int(-1*i - 135)])
        most_changing.append(word_change_mapping[int(-1*i)][0])
    l = open("least_changing","w")
    for w in least_changing:
        l.write(w + ' ')
    l.close()
    m = open("most_changing", "w")
    for w in most_changing:
        m.write(w + ' ')
    m.close()
    output.close()


if __name__ == "__main__":
     main()


