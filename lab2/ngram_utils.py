from urllib.request import urlopen
from bs4 import BeautifulSoup
import csv
import gzip
from urllib.request import urlopen
import uuid
import os
import pandas as pd

def read_ngram_files(filenames):
    for i, fname in enumerate(filenames):

        if i % 20 == 0:
            print(i)

        print('loading file {}'.format(fname))
        gzresp = urlopen(fname)
        unique_filename = "./tmp/" + fname.split('-')[-1].split('.')[0] + "-" + str(uuid.uuid4())
        tempgz = open(unique_filename, "wb")

        '''
        try:
            page = gzresp.read()
        except (http.client.IncompleteRead) as e:
            page = e.partial
        '''
        page = gzresp.read()
        tempgz.write(page)
        tempgz.close()

        with gzip.open(unique_filename, 'rt') as f:
            for line in f:
                line = line.lower()
                row = line.strip().split('\t')
                yield row

        os.remove(unique_filename)
        print('removed file {}'.format(fname))

def get_fiction_filenames():
    url = "http://storage.googleapis.com/books/ngrams/books/datasetsv2.html"
    page = urlopen(url)
    soup = BeautifulSoup(page, "html.parser")

    for h1 in soup.findAll('h1'):
        if h1.text == "English Fiction":
            for x in h1.find_next_siblings():
                if x.b:
                    text = x.b.find(text=True, recursive=False)
                    if text == '5-grams':
                        urls = []
                        for a in x.find_all('a', href=True):
                            if a.text.isalpha():
                                urls.append(a['href'])
                        return(urls)

def convert_panda_to_w2v(pickled_panda_file):
    m = pd.read_pickle(pickled_panda_file)
    filename = pickled_panda_file.replace('.p', '.txt')
    out = open(filename, 'w')
    out.write('{} {}\n'.format(m.shape[0], m.shape[1]))
    m.to_dense().to_csv(out, sep=' ', header=False)
    # save words -> index mapping
    filename = pickled_panda_file.replace('.p', '.indices.p')
    m.index.values.to_pickle()


if __name__ == "__main__":
    urls = get_fiction_filenames()
    print(urls)
    for i,r in enumerate(read_ngram_files(urls)):
        print(r)
        if i > 10000:
            break

