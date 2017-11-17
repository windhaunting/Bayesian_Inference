#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:16:18 2017

@author: fubao
"""

import numpy as np
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt


def plotQuestionFBarPJ2(J2Mean):
    plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot

    x = np.arange(0, len(J2Mean))            #2. algorithm for you homework p
    plt.bar(x, J2Mean)
    plt.title('Bar plot of J2 given alpha, B')

    plt.xlabel("j2 value") #Y-axis label
    plt.ylabel("sample estimation prob J2 given alpha, B") #X-axis label
    #plt.show()
    plt.savefig("../Figures/QuestionFBar.pdf")



def plotQuestionHHistPA(aMeanStore):
    plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot

    plt.hist(aMeanStore, 50, normed=1, facecolor='g', alpha=0.75)
    
    plt.xlabel('alpha value')
    plt.ylabel('frequency of prob alpha given J,B')
    plt.title('Histogram of Prob alpha given J, B with 10,000 iteration sampling')
    plt.axis([0, 1, 0, 3])
    plt.savefig("../Figures/QuestionHHistogram.pdf")

   # plt.show()
 

def plotQuestionJBarProbJ(JMean):
    
    '''plot Question J bar chart of egitht jars 
    '''
    plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot

    x = np.arange(0, len(JMean))            #2. algorithm for you homework p
    plt.bar(x, JMean)
    plt.title('Bar plot of J given alpha, B')

    plt.xlabel("J mean value") #Y-axis label
    plt.ylabel("sample estimation prob J given alpha, B") #X-axis label
    #plt.show()
    plt.savefig("../Figures/QuestionJBarChartJar.pdf")


    plt.show()
    
def plotQuestionJHistPAlpha(aStore):
    
    '''plot Question J histogram alpha 
    '''
    plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot

    plt.hist(aStore, 50, normed=1, facecolor='g', alpha=0.75)
    
    plt.xlabel('alpha value')
    plt.ylabel('frequency of prob alpha given J,B')
    plt.title('Histogram of Prob alpha given J, B with 10,000 iteration sampling')
    #plt.axis([0, 1, 0, 3])
    plt.savefig("../Figures/QuestionJHistogramAlpha.pdf")

    plt.show()
    