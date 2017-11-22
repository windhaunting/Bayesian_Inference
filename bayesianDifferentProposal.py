#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:01:12 2017

@author: fubao
"""

from run_me import getNewAJQuestionI
from run_me import getJointAJBQuestionD

#for extra credit

# extra proposal distribution to test for accerating the sampling converge and kepp the accuracy

# guassian  standard normal function

#original uniform for alpha and flipping for J


def normallFuction():
    
    x = 1


def MCMCATestConvergeTime(BArray, iters):
    '''
    Use Metropolis Hastings algorithm to draw sample from P(J,a|B)
    '''
    error = 0.01
    
    a = 0.01  
    JArray = np.array([0]*len(BArray))       ##initialize JMean

    aMean = a      #initialize JMean
    JMeanArray = JArray #np.array([0]*len(JArray))       ##initialize JMean
    aStore = np.array([0]*iters, dtype=float)
    for i in range(0, iters):
        #propose new alpha and J
        aNew, JArrayNew = getNewAJQuestionI(a, JArray)
        #acceptance ratio
        acceptRatio = getJointAJBQuestionD(JArrayNew, BArray, aNew) / getJointAJBQuestionD(JArray, BArray, a)
        
        if np.random.rand() <= acceptRatio:        #accept new JArray
            a = aNew
            JArray = JArrayNew
            
        
        #mean alpha and JArray
        aMean += a
        JMeanArray += JArray
        aStore[i] = format(a, '.4f')
    
    aMean = aMean / iters
    JMeanArray = np.divide(JMeanArray, iters)
    
    
    return aMean, JMeanArray, aStore



