'''
Check the quality of embeddings.
'''

import itertools
import nltk
import numpy as np
import pandas as pd
import re
import scipy
from collections import Counter, defaultdict
from multiprocessing import Pool
from nltk.corpus import brown, reuters
from nltk.util import ngrams
from tabulate import tabulate
#import pandas as pd
from sklearn.decomposition import TruncatedSVD
from gensim.models.word2vec import Word2Vec

SIM_FILE = "RG_word_sims.tsv"


def test(M, sim_file):
    '''
    Compute Pearson correlation between similarity scores in
    `sim_file` and the cosine similarities in M.

    TODO: what type is M?
    '''

    # assume file is in the format: 'word1 word2 similarity'
    filelines = [line.strip().split() for line in open(sim_file).readlines()]
    P = dict([((w1,w2), float(s)) for (w1,w2,s) in filelines])

    x = []
    y = []

    for (w1, w2) in P.keys():
        try:
            if str(type(M)) == "<class 'pandas.core.frame.DataFrame'>":
                v1 = np.array(M.loc[[w1]])
                v2 = np.array(M.loc[[w2]])
            elif str(type(M)) == "<class 'gensim.models.keyedvectors.KeyedVectors'>" or \
                str(type(M)) == "<class 'gensim.models.word2vec.Word2Vec'>":
                v1 = np.array(M[w1])
                v2 = np.array(M[w2])
            else:
                raise TypeError('Unknown model type', str(type(M)))

        except KeyError:
            continue

        x.append(1 - scipy.spatial.distance.cosine(v1, v2))
        y.append(P[(w1, w2)])

    x = np.array(x)
    y = np.array(y)

    return scipy.stats.pearsonr(x, y)


if __name__ == "__main__":

    pass
