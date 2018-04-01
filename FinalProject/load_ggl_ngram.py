#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 01:11:33 2018

@author: zahra
"""

import os
import time

from_ = 1500
to = 2000

if __name__ == '__main__':
#    with open('../changed_nouns_from_'+str(from_)) as f:
#        content = f.readlines()
#    words = [x.strip() for x in content]
#    
#    print(len(words))
#    for w in words[140:185]:
#        command = 'python ../econpy-google-ngrams/getngrams.py '+w+'_NOUN --startYear='+str(from_)+' --endYear='+str(to)+' --corpus=eng_2012 -caseInsensitive'
#        print(w, command)
#        os.system(command)
#        time.sleep(2.0)
        
#    with open('../unchanged_nouns_from_'+str(from_)) as f:
#        content = f.readlines()
#    words = [x.strip() for x in content]
#    
#    print(len(words))
#    for w in words:
#        command = 'python ../econpy-google-ngrams/getngrams.py '+w+'_NOUN --startYear='+str(from_)+' --endYear='+str(to)+' --corpus=eng_2012 -caseInsensitive'
#        print(w, command)
#        os.system(command)
#        time.sleep(5.0)

    with open('../unchanged_nouns_from_'+str(from_)) as f:
        content = f.readlines()
    words = [x.strip() for x in content]
    print(len(words))
    for w in words:
        with open(w+'_NOUN-eng_2012-1500-2000-3-caseInsensitive.csv') as f:
            content = f.readlines()
        lines = [x.strip() for x in content]
        print(lines[0])
        if len(lines[0]) < 5:
            command = 'python ../econpy-google-ngrams/getngrams.py '+w+'_NOUN --startYear='+str(from_)+' --endYear='+str(to)+' --corpus=eng_2012 -caseInsensitive'
            print(w, command)
            os.system(command)
            time.sleep(3.0)