# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:15:15 2017

@author: Samorg
"""

import math

# calculates the mean
def mean(x):
    sum = 0.0
    for i in x:
         sum += i
    return sum / len(x) 

# calculates the sample standard deviation
def sampleStandardDeviation(x):
    sumv = 0.0
    for i in x:
         sumv += (i - mean(x))**2
    return math.sqrt(sumv/(len(x)-1))

# calculates the PCC using both the 2 functions above
def pearson(x,y):
    scorex = []
    scorey = []

    for i in x: 
        scorex.append((i - mean(x))/sampleStandardDeviation(x)) 

    for j in y:
        scorey.append((j - mean(y))/sampleStandardDeviation(y))

# multiplies both lists together into 1 list (hence zip) and sums the whole list
    if len(x)!=1:
        a=float(sum([i*j for i,j in zip(scorex,scorey)]))/(len(x)-1) 
    else:
        a=0
    return a