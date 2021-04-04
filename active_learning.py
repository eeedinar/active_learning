# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 23:21:21 2021

@author: byzkl
"""
import math
import numpy as np
import numpy.ma as ma

def calc_entropy(predictions,num_of_classes):
    ent = 0
    for i in range(num_of_classes):
        ent -= predictions[i] * math.log(predictions[i],10)
    return ent

# def calc(predictions,num_of_classes):
#     ent = 0
#     for i in range(num_of_classes):
#         ent -= predictions[i] * math.log(predictions[i],10)
#     return ent

def choose_sample(samples,num_of_classes,acquisition,num_of_selection):
    if acquisition=='entropy':
        entropies = {}
        for i, sample in enumerate(samples):
            ent = calc_entropy(sample,num_of_classes)
            entropies[i] = ent
        entropies = sorted(entropies.items(), key=lambda item: item[1], reverse=True)
        entropies = [i[0] for i in entropies]
        entropies = entropies[:num_of_selection]
        print(entropies)
        return entropies
    elif acquisition=='least confident':
        confs = {}
        for i, sample in enumerate(samples):
            conf = np.max(sample)
            confs[i] = conf
        confs = sorted(confs.items(), key=lambda item: item[1])
        confs = [i[0] for i in confs]
        confs = confs[:num_of_selection]
        print(confs)
        return confs
    elif acquisition=='margin sampling':
        margins = {}
        for i, sample in enumerate(samples):
            predictions = np.copy(sample)
            conf1 = np.max(predictions)
            mask = predictions == conf1
            predictions = ma.masked_array(predictions, mask = mask)
            # predictions.remove(conf1)
            conf2 = np.max(predictions)
            print(conf1,conf2)
            margins[i] = conf1-conf2
        margins = sorted(margins.items(), key=lambda item: item[1])
        margins = [i[0] for i in margins]
        margins = margins[:num_of_selection]
        print(margins)
        return margins
   
# samples  = [[0.4,0.6],[0.5,0.5],[0.3,0.7]]    
# # predictions = [0.4,0.6]
# num_of_classes = 2
# # print(calc_entropy(predictions,num_of_classes))
# # choose_sample(samples,num_of_classes,'entropy')
# # choose_sample(samples,num_of_classes,'least confident')
# choose_sample(samples,num_of_classes,'margin sampling')