import gc
from sklearn.decomposition import TruncatedSVD
from collections import Counter, defaultdict
from ngram_utils import get_fiction_filenames, read_ngram_files
from functools import partial
import pandas as pd
import numpy as np
import os
import multiprocessing
import dill as pickle
import random

SIM_FILE = "RG_word_sims.tsv"

def m(ngram_files, word_file, from_, to):
    '''
    window is defined by ngrams corpus passed through `ngram_loc`
    '''

    words = [line.split()[0].lower() for line in open(word_file).readlines()]
    #words = np.random.choice(words, size=25000, replace=False)
    #words = np.random.choice(words, size=10, replace=False)#zahra
    words = frozenset(words)
    print("number of top words: {}".format(len(words)))

    # word1 -> word2 -> cooccurance frequency
    d = defaultdict(lambda : defaultdict(int))

    for i, row in enumerate(read_ngram_files(ngram_files)):

        #if int(row[1]) < from_ or int(row[1]) > to:
        if int(row[1]) < from_ or int(row[1]) > to:
            continue

        ngram = row[0].split()
        match_count = int(row[2])
        t1 = ngram[0].split('_')[0]
        t2 = ngram[-1].split('_')[0]
        if t1 in words:
            for c in ngram[1:]:
                c = c.split('_')[0]
                if c in words:
                    d[t1][c] += match_count
        if t2 in words:
            for c in ngram[:-1]:
                c = c.split('_')[0]
                if c in words:
                    d[t2][c] += match_count

    print(len(d))
    ts = [(k,v) for k,v in d.items()]
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


def m_helper(args):
    return m(*args)


def pmi(M, k=1, normalized=False):
    '''
    If k > 1, compute PMI^k
    (see: "Handling the Impact of Low frequency Events..." (2011))
    '''

    # remove columns and rows that are all 0s
    print('removing all-0 rows and columns')
    M_ = M[(M.T != 0).any()]
    M = M_.T[(M_ != 0).any()].T
    del M_
    gc.collect()

    print('computing P_xy')
    P_xy = M / M.values.sum()

    print('computing P_x')
    P_x = M.sum(axis=0) / M.values.sum()
    pmi = P_xy.div(P_x, axis=1)
    del P_x
    gc.collect()

    print('computing P_y')
    P_y = M.sum(axis=1) / M.values.sum()
    pmi = pmi.div(P_y, axis=0)
    del P_y
    gc.collect()

    print('computing pmi')
    pmi = np.log2(pmi)
    gc.collect()

    if normalized:
        return pmi.div(-np.log(P_xy))
    return pmi


if __name__ == "__main__":
    urls = get_fiction_filenames()
    print("total files: {}".format(len(urls)))
    random.shuffle(urls) # shuffle
    nprocesses = 64

    svd = TruncatedSVD(n_components=500, algorithm="arpack")

    def chunk(xs, n):
        '''Split the list, xs, into n chunks'''
        L = len(xs)
        assert 0 < n <= L
        s = L//n
        return [xs[p:p+s] for p in range(0, L, s)]

    chunks = chunk(urls, nprocesses)

    #for year in range(1800, 2000, 100):
    for year in [1850, 1900, 1950, 2000]:
        word_file = "./top_unigrams_200000/{}".format(str(year+5))
        from_ = year
        to = year + 9
        #'''
        args = []
        for chunk in chunks:
            args.append((chunk, word_file, from_, to))

        with multiprocessing.Pool(processes=nprocesses) as pool:
            ds = pool.map(m_helper, args)
        gc.collect()
        #'''
        #d = m(urls, word_file, from_, to)
        # remove files in tmp folder
        tmp_dir = './tmp'
        for fn in os.listdir(tmp_dir):
            fullpath = os.path.join(tmp_dir, fn)
            os.remove(fullpath)

        # convert returned sets into dicts
        #ds = [dict(d) for d in ds]
        gc.collect()
        #'''
        print("combining dictionaries...")
        d = dict(ds[0])
        del ds[0]
        gc.collect()
        for d_ in ds[1:]:
            d.update(dict(d_))
            del d_
            gc.collect()
        #'''
        #d = dict(d)
        #gc.collect()
        print("creating dataframe...")
        M = pd.DataFrame(d).fillna(0)
        del d
        gc.collect()
        print("computing ppmi")
        M = pmi(M)
        M[M < 0] = 0
        '''
        M = M.to_sparse()
        gc.collect()
        print('pickling...')
        M.to_pickle('./matrices/M.{}.p'.format(str(year)))
        '''

        index = M.index
        print('converting to numpy matrix ...')
        M = M.as_matrix()
        gc.collect()
        print('computing svd...')
        M_500d = svd.fit_transform(M)
        del M
        gc.collect()
        df500 = pd.DataFrame(M_500d, index=index)
        del M_500d
        gc.collect()
        out = open('./matrices/M.{}.500d.txt'.format(year), 'w')
        out.write('{} {}\n'.format(df500.shape[0], df500.shape[1]))
        df500.to_csv(out, sep=' ', header=False)
        out.close()
        del df500
        gc.collect()
