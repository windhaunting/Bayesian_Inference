# Import python modules
import numpy as np
import kaggle
import random

################################################
    
def getJAQuestionA(JArray, a):
    '''
    Implement a function to calculate P (J|α) with input arguments J, α and output P (J|α)
    JArray: a sequence of jar
    a is the model paameter deciding the next jar same as the previous one or not
    '''
 
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
    JArray[index] = 0 if JArray[index] == 1 else 1

    return JArray


def getNewAQuestionG(a):
    
    aNew = np.random.random_sample()
    

def MCMCJQuestionF(BArray, a, iters):
    '''
    Use Metropolis Hastings algorithm to draw sample from P(J|a,B)
    '''
    #aMean = a
    JMeanArray = np.array([0]*len(BArray))       ##initialize JMean
    
    JArray = JMeanArray         #initialize J
    
    for i in range(0, iters):
        #propose new JArray
        
        #acceptance ratio
        acceptRatio = getJointAJBQuestionD(JArrayNew, BArray, a) / getJointAJBQuestionD(JArrayNew, BArray, a)
        
        if np.random.rand(0,1) <= acceptRatio:        #accept new JArray
            JArray = JArrayNew
        
        #mean JArray
        JMeanArray += JArray
    
        #print("JMeanArray: ", JArrayNew)
    
    JMeanArray = np.divide(JMeanArray, iters)
        
    print("JMeanArray: ", JMeanArray, np.mean(JMeanArray))

    PJointAJB = getJointAJBQuestionD(JMeanArray, BArray, a) 
    
    PAB = getJAQuestionA(JMeanArray, a) * getBJQuestionB(JMeanArray, BArray) * getPriorQuestionC(a)
    PJGivenAB = PJointAJB / PAB

    print("PJGivenAB: ", PJGivenAB)


def MCMCJQuestionH(JArray, BArray, iters):
    '''
    Use Metropolis Hastings algorithm to draw sample from P(J|a,B)
    '''
    aMean = 0       #initialize JMean
    
    
    for i in range(0, iters):
        #propose new JArray
        JArrayNew = getNewJQuestionE(JMeanArray)
        #acceptance ratio
        acceptRatio = getJointAJBQuestionD(JArrayNew, BArray, a) / getJointAJBQuestionD(JArrayNew, BArray, a)
        
        if np.random.rand(0,1) <= acceptRatio:        #accept new JArray
            JArray = JArrayNew
        
        #mean JArray
        JMeanArray += JArray
    
        #print("JMeanArray: ", JArrayNew)
    
    JMeanArray = np.divide(JMeanArray, iters)
        
    print("JMeanArray: ", JMeanArray, np.mean(JMeanArray))

    PJointAJB = getJointAJBQuestionD(JMeanArray, BArray, a) 
    
    PAB = getJAQuestionA(JMeanArray, a) * getBJQuestionB(JMeanArray, BArray) * getPriorQuestionC(a)
    PJGivenAB = PJointAJB / PAB

    print("PJGivenAB: ", PJGivenAB)
    
    
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
    JArray = getNewJQuestionE(JArray)
    print ("Flipped JArray: ", JArray)

    '''
    print ("beginning Question 1f")
    BArray = np.array([1,0,0,1,1])
    a = 0.5
    iters = 10000
    MCMCJQuestionF(BArray, a, iters)

'''
	print('1m')
	lengths = [10, 15, 20, 25]
	prediction_prob = list()
	for l in lengths:
		B_array = np.loadtxt('../../Data/B_sequences_%s.txt' % (l), delimiter=',', dtype=float)
		for b in np.arange(B_array.shape[0]):
			prediction_prob.append(np.random.rand(1))
			print('Prob of next entry in ', B_array[b, :], 'is black is', prediction_prob[-1])

	# Output file location
	file_name = '../Predictions/best.csv'

	# Writing output in Kaggle format
	print('Writing output to ', file_name)
	kaggle.kaggleize(np.array(prediction_prob), file_name)


'''