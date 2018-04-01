#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 19:13:28 2018

@author: zahra
"""

import json

if __name__ == '__main__':
    with open('BNC_word_list') as f:
        content = f.readlines()
    words = [x.strip() for x in content]
#    print(type(words), len(words))
    freq_map = {}
    for w in words:
        w = w.strip().split()
        if w[2] in freq_map:
            if w[3] == 'n':
                freq_map[w[2]] = (float(w[1]), freq_map[w[2]][1], float(w[1])+freq_map[w[2]][1])
            if w[3] == 'v':
                freq_map[w[2]] = (freq_map[w[2]][0], float(w[1]), float(w[1])+freq_map[w[2]][1])
            if w[3] != 'n' and w[3] != 'v':
                freq_map[w[2]] = (0,0, float(w[1])+freq_map[w[2]][2])
        else:
            if w[3] == 'n':
                freq_map[w[2]] = (float(w[1]), 0, float(w[1]))
            if w[3] == 'v':
                freq_map[w[2]] = (0, float(w[1]), float(w[1]))
            if w[3] != 'n' and w[3] != 'v':
                freq_map[w[2]] = (0,0, float(w[1]))
    for w,k in freq_map.items():
        print(w, k)
        
    with open('target_words_freq_n_v_tot', 'w') as file:
        file.write(json.dumps(freq_map))
    file.close()