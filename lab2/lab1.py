from collections import Counter, defaultdict
from ngram_utils import get_fiction_filenames, read_ngram_files
from functools import partial
import pandas as pd
import numpy as np
import multiprocessing
import random

SIM_FILE = "RG_word_sims.tsv"

def m(ngram_files, word_file):
    '''
    window is defined by ngrams corpus passed through `ngram_loc`
    '''

    words = [line.split()[0] for line in open(word_file).readlines()]
    words = np.random.choice(words, size=20000, replace=False)
    words = frozenset(words)

    # word1 -> word2 -> cooccurance frequency
    d = defaultdict(lambda : defaultdict(int))

    for i, row in enumerate(read_ngram_files(ngram_files)):

        if row[1] != "1985":
            continue

        ngram = row[0].split()
        match_count = int(row[2])
        t1 = ngram[0]
        t2 = ngram[-1]
        if t1 in words:
            for c in ngram[1:]:
                if c in words:
                    d[t1][c] += match_count
        if t2 in words:
            for c in ngram[:-1]:
                if c in words:
                    d[t2][c] += match_count

    #M = pd.DataFrame(d).fillna(0).to_sparse()
    #print(M)
    #M.to_pickle('M.1985.p')
    ts = []
    for w1, dic in d.items():
        t = (w1, [(k,v) for k,v in dic.items()])
        ts.append(t)
    return ts

    '''
    W = list(dict(freq_dict.most_common(n)).keys())
    if include_words:
        W.extend(include_words)

    # Get bigrams from words in W
    corpora_iter = itertools.chain.from_iterable([c.words() for c in corpora])
    bigrams = ngrams([w.lower() for w in corpora_iter], 2)
    # filter out words not in W
    bigrams = [bi for bi in bigrams if bi[0] in W and bi[1] in W]

    # convert bigram counts to cooccurance matrix
    c = Counter(bigrams)

    if preceeding:
        # reverse bigram tuples
        bigrams_reversed = [(t[1], t[0]) for t in bigrams]
        c2 = Counter(bigrams_reversed)
        c.update(c2)

    d = defaultdict(lambda : defaultdict(int))
    for t,freq in c.items():
        d[t[1]][t[0]] = freq

    M1 = pd.DataFrame(d).fillna(0).to_sparse()

    # remove words not in both rows and columns
    M1 = M1.drop(list(set(M1.index) - set(M1.columns.values)))
    M1 = M1.drop(list(set(M1.columns.values) - set(M1.index)), axis=1)

    return M1
    '''

def pmi(M, k=1, normalized=False):
    '''
    If k > 1, compute PMI^k
    (see: "Handling the Impact of Low frequency Events..." (2011))
    '''
    # +1 for smoothing
    P_xy = (M + 1) / M.values.sum()
    P_xy = P_xy.pow(k)
    P_x = (M.sum(axis=1) + 1) / M.values.sum()
    P_y = (M.sum(axis=0) + 1) / M.values.sum()
    pmi = np.log(P_xy.div(P_x, axis=1).div(P_y, axis=0))
    if normalized:
        return pmi.div(-np.log(P_xy))
    return pmi


if __name__ == "__main__":
    urls = get_fiction_filenames()
    random.shuffle(urls)
    word_file = "./top_unigrams_50000/1985"

    nprocesses = 4
    chunks = [urls[i:i+nprocesses] for i in range(0, len(urls), nprocesses)]
    with multiprocessing.Pool(processes=nprocesses) as pool:
        m_ = partial(m, word_file=word_file)
        ds = pool.map(m_, chunks)

    ds_ = []
    for d in ds:
        d_ = defaultdict(lambda : defaultdict(int))
        for k,v in d:
            d_[k] = dict(v)
        ds_.append(d_)

    d = ds_[0]
    for d_ in ds_[1:]:
        d.update(d_)

    M = pd.DataFrame(d).fillna(0).to_sparse()
    M.to_pickle('M.1985.p')

