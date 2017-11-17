# Import python modules
'''
@author: fubao
'''

import numpy as np
import kaggle
import random

from plotting import plotQuestionFBarPJ2
from plotting import  plotQuestionHHistPA
from plotting import plotQuestionJBarProbJ
from plotting import plotQuestionJHistPAlpha
from plotting import plotQuestionJAlphaIteration

################################################
    
def getJAQuestionA(JArray, a):
    '''
    Implement a function to calculate P (J|α) with input arguments J, α and output P (J|α)
    JArray: a sequence of jar
    a is the model paameter deciding the next jar same as the previous one or not
    '''
    if JArray[0] == 1:
        return 0.0
    PJGivenA = 1          #prob of jar sequence given  \alpha
    for i in range(1, len(JArray)):
        PJGivenA = PJGivenA * a if JArray[i] == JArray[i-1] else PJGivenA * (1-a)
                   
    return PJGivenA

def getBJQuestionB(JArray, BArray):
    '''
    Implement a function to calculate P (BjJ) with input arguments J, B and output P (BjJ).
    ''' 
    PBGivenJ = 1               # prob of Ball sequences given Jar sequences
    for i in range(0, len(JArray)):
        if JArray[i] == 0:
            PBGivenJ = PBGivenJ * 0.2 if BArray[i]== 0 else PBGivenJ * 0.8
        elif JArray[i] == 1:
            PBGivenJ = PBGivenJ * 0.9 if BArray[i]== 0 else PBGivenJ * 0.1
    
    return PBGivenJ

def getPriorQuestionC(a):
    '''
    Implement a function to calculate P(α) with input argument α and output P(α
    get prior probability p(a)
    '''
    PA  =0       #pa, probability of \alpha
    if a >= 0 and a <= 1:
        PA = 1
    
    return PA


    
def getJointAJBQuestionD(JArray, BArray, a):
    '''
    Implement a function to calculate P(α; J; B) with input arguments J, B, α and output
    P(α; J; B).
    '''
    PJGivenA = getJAQuestionA(JArray, a)
    PBGivenJ = getBJQuestionB(JArray, BArray)
    PA = getPriorQuestionC(a)
    
    PJointAJB = PJGivenA * PBGivenJ * PA           #joint probablity of P(α, J, B).

    return PJointAJB


def getNewJQuestionE(JArray):
    '''
    a function to generate a new proposed value for J with input argument J and output
    Jnew. This is calculated by randomly selecting a Ji and flipping its value.
    '''
    index = random.randint(0,len(JArray)-1)
    JArrayNew = JArray.copy()
    JArrayNew[index] = 0 if JArrayNew[index] == 1 else 1
     
    return JArrayNew



def MCMCJQuestionF(BArray, a, iters):
    '''
    Use Metropolis Hastings algorithm to draw sample from P(J|a,B)
    '''
    #aMean = a
    JMeanArray = np.array([0]*len(BArray))       ##initialize JMean
    
    JArray = JMeanArray         #initialize J

    for i in range(0, iters):
        #propose new JArray
        JArrayNew = getNewJQuestionE(JArray)
        #acceptance ratio
        acceptRatio =  getJointAJBQuestionD(JArrayNew, BArray, a) / getJointAJBQuestionD(JArray, BArray, a)
        if np.random.rand() <= acceptRatio:        #accept new JArray
            JArray = JArrayNew

        #mean JArray
        JMeanArray += JArray
        
        #print("J2MeanStoresArray: ", JMeanArray[1],i, J2MeanStoresArray[i], JMeanArray[1]/ (i+1))
    JMeanArray = np.divide(JMeanArray, iters)
        
    #print("JMeanArray: ", JMeanArray)

    return JMeanArray


def getNewAQuestionG(a):
    '''
    a new proposed value for α with input argument α and output
    αnew. This is calculated by selecting a new α value uniformly at random
    '''
    aNew = np.random.random_sample()
    return aNew


def MCMCAQuestionH(JArray, BArray, iters):
    '''
    Use Metropolis Hastings algorithm to draw sample from P(J|a,B)
    '''
    aMean = 0.01       #initialize JMean
    a = aMean
    aStore = np.array([0]*iters, dtype=float)
    for i in range(0, iters):
        #propose new alpha
        aNew = getNewAQuestionG(a)
        #acceptance ratio
        acceptRatio = getJointAJBQuestionD(JArray, BArray, aNew) / getJointAJBQuestionD(JArray, BArray, a)
        if np.random.rand() <= acceptRatio:        #accept new JArray
            a = aNew
        #mean alpha
        aMean += a
        aStore[i] = format(a, '.4f')
        
    aMean = aMean / iters
    return aMean, aStore


def getNewAJQuestionI(a, JArray):
    '''
    function to generate a proposed values for α and J with input argument α, J and output αnew, Jnew. 
    This is performed by invoking the proposal functions for α (1g) and J (1e) independently
    '''
    aNew = getNewAQuestionG(a)
    JArrayNew = getNewJQuestionE(JArray)
    
    return aNew, JArrayNew
 
    
def MCMCAJQuestionJ(BArray, iters):
    '''
    Use Metropolis Hastings algorithm to draw sample from P(J,a|B)
    '''
    
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


def getNextBallQuestionK(Jn, a):
    '''
    return the probability of a black ball in N + 1th jar given the Nth jar
    and α only.

    '''
    
    pBnext = 0               #the n+1th ball given Jn and alpha
    if Jn == 0:            #Jar 0 
        pBnext = a*0.8 + (1-a)*0.1
    elif Jn == 1:          
        pBnext = a*0.1 + (1-a)*0.8
       
    return pBnext
    
'''
def getNextBallMCMCQuestionl(BArray, iters):
    
    aMean, JMeanArray, aStore = MCMCAJQuestionJ(BArray, iters)
    print ("JMeanArray : ", aMean, JMeanArray)
    JMean = JMeanArray[-1]
    Jn = 1
    if JMean >= 0.5:
        Jn = 1
    else:
        Jn = 0
    print ("Jn : ", Jn)
    pBnext = getNextBallQuestionK(Jn, aMean)
    
    return pBnext
'''

def getNextBallMCMCQuestionl(BArray, iters):

    a = 0.01  
    JArray = np.array([0]*len(BArray))       ##initialize JMean
    prob = 0
    for i in range(0, iters):
        #propose new alpha and J
        aNew, JArrayNew = getNewAJQuestionI(a, JArray)
        #acceptance ratio
        acceptRatio = getJointAJBQuestionD(JArrayNew, BArray, aNew) / getJointAJBQuestionD(JArray, BArray, a)
        if np.random.rand() <= acceptRatio:        #accept new JArray
            a = aNew
            JArray = JArrayNew
            #mean alpha and JArray
        if JArray[-1] >= 0.5:
            Jn = 1
        else:
            Jn = 0
        prob += getNextBallQuestionK(Jn, a)
    return prob/iters       
        


def predictNextBallBayesInference(file_name):
    iterations = np.arange(10000, 1000000, 10000) #[100000] #np.arange(10000, 1000000, 10000)
    for iters in iterations:
        prediction_prob = list()
        lengths = [10, 15, 20, 25]
        for l in lengths:
            BArray = np.loadtxt('../../Data/B_sequences_%s.txt' % (l), delimiter=',', dtype=float)
            for b in np.arange(BArray.shape[0]):
                prob = getNextBallMCMCQuestionl(BArray[b, :], iters)
                prediction_prob.append(prob)
                #print('Prob of next entry in ', BArray[b, :], 'is black is', prediction_prob[-1])
                #print('Prob of next entry in is black is', prediction_prob[-1])
        print('Writing output to ', file_name + "_iteration_" + str(iters))
        kaggle.kaggleize(np.array(prediction_prob), file_name + "iteration_" + str(iters))
    
if __name__== "__main__":

    ################################################
    print('1a through 1l computation goes here ...')
    
    '''
    print ("beginning Question 1a")
    JArray = np.array([0, 1, 1, 0, 1])
    a = 0.75
    PJGivenA = getJAQuestionA(JArray, a)
    print ("PJGivenA: ", PJGivenA)

    JArray = np.array([0, 0, 1, 0, 1])
    a = 0.2
    PJGivenA = getJAQuestionA(JArray, a)
    print ("PJGivenA: ", PJGivenA)

    JArray = np.array([1,1,0,1,0,1])
    a = 0.2
    PJGivenA = getJAQuestionA(JArray, a)
    print ("PJGivenA: ", PJGivenA)

    JArray = np.array([0,1,0,1,0,0])
    a = 0.2
    PJGivenA = getJAQuestionA(JArray, a)
    print ("PJGivenA: ", PJGivenA)
    
    
    print ("beginning Question 1b")
    JArray = np.array([0,1,1,0,1])
    BArray = np.array([1,0,0,1,1])
    PBGivenJ = getBJQuestionB(JArray, BArray)
    print ("PBGivenJ: ", PBGivenJ)

    JArray = np.array([0,1,0,0,1])
    BArray = np.array([0,0,1,0,1])
    PBGivenJ = getBJQuestionB(JArray, BArray)
    print ("PBGivenJ: ", PBGivenJ)

    JArray = np.array([0,1,1,0,0,1])
    BArray = np.array([1,0,1,1,1,0])
    PBGivenJ = getBJQuestionB(JArray, BArray)
    print ("PBGivenJ: ", PBGivenJ)

    JArray = np.array([1,1,0,0,1,1])
    BArray = np.array([0,1,1,0,1,1])
    PBGivenJ = getBJQuestionB(JArray, BArray)
    print ("PBGivenJ: ", PBGivenJ)
    
    
    print ("beginning Question 1c")
    PA = getPriorQuestionC(2)
    print ("PA: ", PA)
    
    
    print ("beginning Question 1d")
    
    JArray = np.array([0,1,1,0,1])
    BArray = np.array([1,0,0,1,1])
    a = 0.75
    PJointAJB = getJointAJBQuestionD(JArray, BArray, a)
    print ("Joint P(α, J, B) PJointAJB: ", PJointAJB)

    JArray = np.array([0,1,0,0,1])
    BArray = np.array([0,0,1,0,1])
    a = 0.3
    PJointAJB = getJointAJBQuestionD(JArray, BArray, a)
    print ("Joint P(α, J, B) PJointAJB: ", PJointAJB)
    
    JArray = np.array([0,0,0,0,0,1])
    BArray = [0,1,1,1,0,1]
    a = 0.63
    PJointAJB = getJointAJBQuestionD(JArray, BArray, a)
    print ("Joint P(α, J, B) PJointAJB: ", PJointAJB)

    JArray = np.array([0,0,1,0,0,1,1])
    BArray = np.array([1,1,0,0,1,1,1])
    a = 0.23
    PJointAJB = getJointAJBQuestionD(JArray, BArray, a)
    print ("Joint P(α, J, B) PJointAJB: ", PJointAJB)
    
    
    print ("beginning Question 1e")
    JArray = np.array([1,1,0])
    JArrayNew = getNewJQuestionE(JArray)
    print ("Flipped JArrayNew: ", JArrayNew)

    
    
    print ("beginning Question 1f")
    BArray = np.array([1,0,0,1,1])
    a = 0.5
    iters = 10000
    JMeanArray = MCMCJQuestionF(BArray, a, iters)
    print (" JMeanArray: ", JMeanArray)
    
    plotQuestionFBarPJ2([JMeanArray[2], 1-JMeanArray[2]])

    
    BArray = np.array([1,0,0,0,1,0,1,1])
    a = 0.11
    iters = 10000
    JMeanArray = MCMCJQuestionF(BArray, a, iters)
    print (" JMeanArray: ", JMeanArray)
    
    BArray = np.array([1,0,0,1,1,0,0])
    a = 0.75
    iters = 10000
    JMeanArray = MCMCJQuestionF(BArray, a, iters)
    print (" JMeanArray: ", JMeanArray)
       
    
    
    print ("beginning Question 1g")
    a = 0.5
    aNew = getNewAQuestionG(a)
    print ("proposed aNew: ", aNew)
    
    
    
    print ("beginning Question 1h")
    JArray= np.array([0,1,0,1,0])
    BArray = np.array([1,0,1,0,1])
    iters = 10000
    aMean, aMeanStore = MCMCAQuestionH(JArray, BArray, iters)
    print("aMean: ", aMean)

    JArray = np.array([0,0,0,0,0])
    BArray = np.array([1,1,1,1,1])
    iters = 10000   #10000
    aMean, aMeanStore = MCMCAQuestionH(JArray, BArray, iters)
    print("aMean: ", aMean)
    
    JArray = np.array([0,1,1,0,1])
    BArray = np.array([1,0,0,1,1])
    iters = 10000   #10000
    aMean, aStore = MCMCAQuestionH(JArray, BArray, iters)
    print("aMean: ", aMean)
    plotQuestionHHistPA(aStore)
    
    JArray = np.array([0,1,1,1,1,1,1,0])
    BArray = np.array([1,0,0,1,1,0,0,1])
    iters = 10000   #10000
    aMean, aMeanStore = MCMCAQuestionH(JArray, BArray, iters)
    print("aMeanStore: ", aMean, aMeanStore)
    
    JArray = np.array([0,1,1,0,1,0])
    BArray = np.array([1,0,0,1,1,1])
    iters = 10000   #10000
    aMean, aMeanStore = MCMCAQuestionH(JArray, BArray, iters)
    print("aMeanStore: ", aMean, aMeanStore)
    
    
    '''
    
    print ("beginning Question 1j")
             
    BArray = np.array([1,1,0,1,1,0,0,0])
    iters = 10000
    aMean, JMeanArray, aStore = MCMCAJQuestionJ( BArray, iters)
    print("QuestoinJ aMean: ", aMean)
    print("QuestionJ JMeanArray: ", JMeanArray)
    
    #plotQuestionJBarProbJ(JMeanArray)
    #plotQuestionJHistPAlpha(aStore)
    plotQuestionJAlphaIteration(aStore)
    
    '''
    print ("beginning Question 1k")
    a = 0.6
    Jn = 1
    pBnext = getNextBallQuestionK(Jn, a)
    print ("Question k pBnext : ", pBnext)
    
    a = 0.99
    Jn = 0
    pBnext = getNextBallQuestionK(Jn, a)
    print ("Question k pBnext : ", pBnext)
    
    a = 0.33456
    Jn = 0
    pBnext = getNextBallQuestionK(Jn, a)
    print ("Question k pBnext : ", pBnext)    
    
    a = 0.5019
    Jn = 0
    pBnext = getNextBallQuestionK(Jn, a)
    print ("Question k pBnext : ", pBnext) 


    print ("beginning Question 1l")
    BArray = np.array([0, 0, 1])
    iters = 10000
    pBnext = getNextBallMCMCQuestionl(BArray, iters)
    print ("Question l pBnext : ", pBnext) 
    
    BArray = np.array([0, 1, 0, 1, 0, 1])
    iters = 10000
    pBnext = getNextBallMCMCQuestionl(BArray, iters)
    print ("Question l pBnext : ", pBnext) 
    
    BArray = np.array([0, 1, 0, 0, 0, 0, 0])
    iters = 10000
    pBnext = getNextBallMCMCQuestionl(BArray, iters)
    print ("Question l pBnext : ", pBnext) 

    BArray = np.array([1, 1, 1, 1, 1])
    iters = 10000
    pBnext = getNextBallMCMCQuestionl(BArray, iters)
    print ("Question l pBnext : ", pBnext) 
    
    
    print ("beginning Question 1m")
    file_name = '../Predictions/best.csv'
    predictNextBallBayesInference(file_name)
    
    '''
	