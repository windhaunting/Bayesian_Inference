#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:01:12 2017

@author: fubao
"""

from run_me import getNewAJQuestionI
from run_me import getNewJQuestionE
from run_me import getJointAJBQuestionD
from run_me import getNewAQuestionG
import numpy as np

import time

#for extra credit

# extra proposal distribution to test for accerating the sampling converge and kepp the accuracy

# guassian  standard normal function

#original uniform for alpha and flipping for J



def normallFuctionAlpha(a):
    '''
    use normal distribution sampling for alpha
    '''
    newAlpha = np.random.normal(0, 1, None)

    return newAlpha


def normalFunctionGamma(a):
    '''
    use gamma function for alpha
    '''
    newAlpha = np.random.gamma(1, 1, None)
    return newAlpha

def MCMCATestConvergeTime(BArray, iters):
    '''
    Use Metropolis Hastings algorithm to draw sample from P(J,a|B)
    '''
    error = 1e-8
    
    a = 0.01  
    JArray = np.array([0]*len(BArray))       ##initialize JMean

    aMean = a      #initialize JMean
    JMeanArray = JArray #np.array([0]*len(JArray))       ##initialize JMean
    aStore = np.array([0]*iters, dtype=float)
    beginTime = time.time()
    preAMean = 0
    for i in range(0, iters):
        #propose new alpha and J
        aNew = getNewAQuestionG(a)
        #aNew = normallFuctionAlpha(a)
        #aNew = normalFunctionGamma(a)
        JArrayNew = getNewJQuestionE(JArray)
        #acceptance ratio
        acceptRatio = getJointAJBQuestionD(JArrayNew, BArray, aNew) / getJointAJBQuestionD(JArray, BArray, a)
        
        if np.random.rand() <= acceptRatio:        #accept new JArray
            a = aNew
            JArray = JArrayNew
        
        #mean alpha and JArray
        aMean += a
        JMeanArray += JArray
        aStore[i] = format(a, '.4f')
        
        if (abs(aMean/(i+1) - preAMean)) < error:      #converge
            print ("End time ",aMean/(i+1), preAMean, error, time.time()-beginTime, i)   
            break
        preAMean = aMean/(i+1)
       
        
    aMean = aMean / iters
    JMeanArray = np.divide(JMeanArray, iters)
    
    
    return aMean, JMeanArray, aStore


if __name__== "__main__":

    BArray = np.array([1,1,0,1,1,0,0,0])
    iters = 100000
    aMean, JMeanArray, aStore = MCMCATestConvergeTime( BArray, iters)
    print("QuestoinJ aMean: ", aMean)
    print("QuestionJ JMeanArray: ", JMeanArray)