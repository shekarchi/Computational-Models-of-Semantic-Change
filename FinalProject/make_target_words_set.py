#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 00:43:08 2018

@author: zahra
"""

import json

from_ = 1500
to_ = 2000

if __name__ == '__main__':
    fp = open('BNC_words_date_type_map_1.txt', 'r')
    BNC_date_type_map = json.load(fp)
    print(len(BNC_date_type_map))

    fp = open('BNC_words_date_type_map_2.txt', 'r')
    BNC_date_type_map_t = json.load(fp)
    print(len(BNC_date_type_map_t))

    BNC_date_type_map.update(BNC_date_type_map_t)

    fp = open('BNC_words_date_type_map_3.txt', 'r')
    BNC_date_type_map_t = json.load(fp)
    print(len(BNC_date_type_map_t))

    BNC_date_type_map.update(BNC_date_type_map_t)

    fp = open('BNC_words_date_type_map_4.txt', 'r')
    BNC_date_type_map_t = json.load(fp)
    print(len(BNC_date_type_map_t))

    BNC_date_type_map.update(BNC_date_type_map_t)

    fp = open('BNC_words_date_type_map_5.txt', 'r')
    BNC_date_type_map_t = json.load(fp)
    print(len(BNC_date_type_map_t))

    BNC_date_type_map.update(BNC_date_type_map_t)
    
    fp = open('BNC_words_date_type_map_6.txt', 'r')
    BNC_date_type_map_t = json.load(fp)
    print(len(BNC_date_type_map_t))

    BNC_date_type_map.update(BNC_date_type_map_t)

    '''...'''


    BNC_date_type_cat = {}
    for w, data in BNC_date_type_map.items():
#        print(w, len(data), data)
#        if len(data) == 0:
#            continue
        new_ = True
        noun = False
        for d in data:
            if d[0]<from_ or d[0]>to_:
                new_ = False
            if d[1] == 'n.':
                noun = True
        if new_ and noun:
            BNC_date_type_cat[w] = data
    print('len(BNC_date_type_cat):',len(BNC_date_type_cat))
#    print(BNC_date_type_cat)


    with open('BNC_target_words.txt', 'w') as file:
        file.write(json.dumps(BNC_date_type_cat))
    file.close()
