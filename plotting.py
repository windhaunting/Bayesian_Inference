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


def plotExample():
    #Create values and labels for bar chart
    values =np.random.rand(3)
    inds   =np.arange(3)
    labels = ["A","B","C"]
    
    #Plot a bar chart
    plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.bar(inds, values, align='center') #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel("Error") #Y-axis label
    plt.xlabel("Method") #X-axis label
    plt.title("Error vs Method") #Plot title
    plt.xlim(-0.5,2.5) #set x axis range
    plt.ylim(0,1) #Set yaxis range
    
    #Set the bar labels
    plt.gca().set_xticks(inds) #label locations
    plt.gca().set_xticklabels(labels) #label values
    
    #Save the chart
    plt.savefig("../Figures/example_bar_chart.pdf")
    
    #Create values and labels for line graphs
    values =np.random.rand(2,5)
    inds   =np.arange(5)
    labels =["Method A","Method B"]
    
    #Plot a line graph
    plt.figure(2, figsize=(6,4))      #6x4 is the aspect ratio for the plot
    plt.plot(inds,values[0,:],'or-', linewidth=3) #Plot the first series in red with circle marker
    plt.plot(inds,values[1,:],'sb-', linewidth=3) #Plot the first series in blue with square marker
    
    #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel("Error") #Y-axis label
    plt.xlabel("Value") #X-axis label
    plt.title("Error vs Value") #Plot title
    plt.xlim(-0.1,4.1) #set x axis range
    plt.ylim(0,1) #Set yaxis range
    plt.legend(labels,loc="best")
    
    #Save the chart
    plt.savefig("../Figures/example_line_plot.pdf")
    
    #Displays the plots.
    #You must close the plot window for the code following each show()
    #to continue to run
    plt.show()


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
    plt.ylabel('Prob alpha given J,B')
    plt.title('Histogram of Prob alpha given J, B with 10,000 iteration sampling')
    plt.axis([0, 1, 0, 0.03])
    plt.savefig("../Figures/QuestionHHistogram.pdf")

    plt.show()