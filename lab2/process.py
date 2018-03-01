import csv
import gzip
import heapq
import os
import string
from urllib.request import urlopen
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def main():
    ngrams_loc = "http://storage.googleapis.com/books/ngrams/books/"
    unigram_filename_template = 'googlebooks-eng-fiction-all-1gram-20120701-{}.gz'
    unigram_filenames = [ngrams_loc + unigram_filename_template.format(let) \
            for let in string.ascii_lowercase]
    top_n_unigrams_per_decade(100000, unigram_filenames, save_to_dir='top_unigrams_100000')


def top_n_unigrams_per_decade(n, filenames, from_=1800, to=2010,  save_to_dir=False):
    '''
    Return a list of the top `n` unigrams in `filenames`
    with their frequencies, per decade.
    '''

    if save_to_dir:
        # make directory to store top words
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)

    hs = {} # store top n words, per decade, in a heap
    for x in range(from_, to, 10):
        hs[x+5] = []

    for fname in filenames:
        print('loading file {}'.format(fname))
        gzresp = urlopen(fname)
        tempgz = open("/tmp/tempfile.gz", "wb")
        tempgz.write(gzresp.read())
        tempgz.close()

        with gzip.open("/tmp/tempfile.gz", 'rt') as f:
            fieldnames=['unigram', 'year', 'match_count', 'volume_count']
            reader = csv.DictReader(f, delimiter='\t', fieldnames=fieldnames)
            for row in reader:

                year = int(row['year'])
                if year < from_ or to < year:
                    continue
                # compute most frequent for the "middle" year
                if year % 10 != 5:
                    continue

                unigram = row['unigram'].lower().split('_')[0]
                if unigram in STOPWORDS:
                    # ignore stopwords
                    continue

                count = int(row["match_count"]) # incorporate volume count?
                t = (count, row['unigram'])
                if len(hs[year]) < n:
                    heapq.heappush(hs[year], t)
                elif count > hs[year][0][0]:
                    heapq.heappushpop(hs[year], t)

    for y, top in hs.items():
        filename = os.path.join(save_to_dir, str(y))
        with open(filename, 'w') as f:
            for count, unigram in top:
                f.write("{}\t{}\n".format(unigram, str(count)))


if __name__ == "__main__":
    main()

