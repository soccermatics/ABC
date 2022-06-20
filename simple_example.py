#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:04:15 2022

@author: davsu428
"""


import numpy as np

cat_scores = np.array([79, 85, 82, 68, 87, 66])
noncat_scores = np.array([61, 39, 28, 73, 74, 12])

threshold = 60

real_cats = cat_scores
predicted_cats =np.append(cat_scores[cat_scores>=threshold],noncat_scores[noncat_scores>=threshold])

sample_size=1000000
real_rand=np.random.choice(real_cats,sample_size)
predicted_rand=np.random.choice(predicted_cats,sample_size)

not_equal=np.not_equal(real_rand,predicted_rand)
real_rand=real_rand[not_equal]
predicted_rand=predicted_rand[not_equal]

print("Probability real scores more than predicted. B(r)")
print(np.sum(real_rand>predicted_rand)/len(real_rand))