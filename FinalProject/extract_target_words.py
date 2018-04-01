#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 23:53:12 2018

@author: zahra
"""

import re
import time
import urllib.request
from bs4 import BeautifulSoup
import json

def load_BNC_words(filepath):
    with open(filepath) as f:
        content = f.readlines()
    words = [x.strip() for x in content]
    return words


DATE_PATTERN = re.compile("\([a-z]*([0-9]{4})")

def parse_word_type(string):
    elem = string.strip().split()
    return elem[2][:-7],elem[1][14:] #word type and categoric id

def parse_word_html(word):
    date_type_map = []
    url = "http://historicalthesaurus.arts.gla.ac.uk/category-selection/?qsearch={}".format(word)
    time.sleep(3)
    response = urllib.request.urlopen(url)
    page = response.read()
    soup = BeautifulSoup(page, 'html.parser')
    mainInner = soup.find('div', {"id": "mainInner"})
    for p in mainInner.find_all('p', {"class": ['catEven', 'catOdd']}, recursive=False):
        word_type = p.find_all('span', {"class": ['small']}, recursive=False)[0]
        print(w, word_type)
        word_type, cID = parse_word_type(str(word_type))
        text = p.find_all(text=True, recursive=False)
        match = DATE_PATTERN.search(text[-1])
        if not match:
            continue
        date = int(match.group(1))
        date_type_map.append((date,word_type, cID))
#        print(word+': ',date,word_type, cID)
    return date_type_map

if __name__ == '__main__':
    print ('hello')
    BNC_words = load_BNC_words('./BNC_words')
#    os.system('python ./econpy-google-ngrams/getngrams.py '+\
#              BNC_words[10] + ' --startYear=1900 --endYear=2000 --corpus=eng_2012 -caseInsensitive')
    print (len(BNC_words))
    BNC_date_type_map = {}
#    for w in BNC_words[:1264]:
#        dtmap = parse_word_html(w)
#        BNC_date_type_map[w] = dtmap
#    with open('BNC_words_date_type_map_1.txt', 'w') as file:
#        file.write(json.dumps(BNC_date_type_map))
#    file.close()
#    time.sleep(600.0)
#    for w in BNC_words[1264:2400]:
#        dtmap = parse_word_html(w)
#        BNC_date_type_map[w] = dtmap
#    with open('BNC_words_date_type_map_2.txt', 'w') as file:
#        file.write(json.dumps(BNC_date_type_map))
#    file.close()
#    time.sleep(600.0)
#    for w in BNC_words[2400:3600]:
#        dtmap = parse_word_html(w)
#        BNC_date_type_map[w] = dtmap
#    with open('BNC_words_date_type_map_3.txt', 'w') as file:
#        file.write(json.dumps(BNC_date_type_map))
#    file.close()
#    time.sleep(1800.0)

#    BNC_date_type_map = {}
#    for w in BNC_words[3600:4800]:
#        dtmap = parse_word_html(w)
#        BNC_date_type_map[w] = dtmap
#    with open('BNC_words_date_type_map_4.txt', 'w') as file:
#        file.write(json.dumps(BNC_date_type_map))
#    file.close()
#    time.sleep(1800.0)
#
#    BNC_date_type_map = {}
#    for w in BNC_words[4800:6000]:
#        dtmap = parse_word_html(w)
#        BNC_date_type_map[w] = dtmap
#    with open('BNC_words_date_type_map_5.txt', 'w') as file:
#        file.write(json.dumps(BNC_date_type_map))
#    file.close()
#    time.sleep(1800.0)
#
    BNC_date_type_map = {}
    for w in BNC_words[6000]:
        dtmap = parse_word_html(w)
        BNC_date_type_map[w] = dtmap
    with open('BNC_words_date_type_map_6.txt', 'w') as file:
        file.write(json.dumps(BNC_date_type_map))
    file.close()
#
#    with open('BNC_words_date_type_map.txt', 'w') as file:
#        file.write(json.dumps(BNC_date_type_map))
#    fp = open('BNC_words_date_type_map.txt', 'r')
#    what = json.load(fp)
#    print(type(what), what['abnormal'])








