#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:43:10 2018

@author: zahra
"""

import json

if __name__ == '__main__':
    fp = open('BNC_target_words.txt', 'r')
    BNC_date_type_cat = json.load(fp)
#    print(len(BNC_date_type_cat))
    w_d = {}
    counter = 0
    for w, data in BNC_date_type_cat.items():
        noun_time = 2018 #now :D
        verb_time = 2018 #now :D
        normal_flag = True
        for d in data:
            if len(d[1]) ==0:
                counter += 1
                normal_flag = False
                break
            if d[1]=='n.':
                if d[0]<noun_time:
                    noun_time = d[0]
#            print (w, d)
            if d[1][0]=='v':
                if d[0]<verb_time:
                    verb_time = d[0]
        if noun_time != 2018 and verb_time != 2018:
            d = verb_time - noun_time
#            counter += 1
        if noun_time == 2018 and verb_time == 2018:
            d = -1.5 #adj or adv
        if noun_time == 2018 and verb_time != 2018:
            #only verb
            d = -2.5
        if noun_time != 2018 and verb_time == 2018:
            #only noun
            d = -3.5
        if d>0:
#            counter += 1
            print(len(w))
        sense_num = 0
        for dt in data:
#            if dt[1]=='n.' and dt[0]<verb_time:
            if dt[0]<verb_time:
                sense_num += 1
        if normal_flag:
            w_d[w] = (noun_time,verb_time, d, len(w), sense_num)
#        print('w:',w, 'len:',len(data), 'nt:', noun_time, 'vt:',verb_time, 'sn:',sense_num, data)
    
    print(len(w_d), counter)
    
#    for key, value in w_d.items():
#        print(key, value)
    
    with open('target_words_n_v_d_len', 'w') as file:
        file.write(json.dumps(w_d))
    file.close()