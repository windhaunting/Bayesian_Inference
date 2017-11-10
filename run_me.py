# Import python modules
import numpy as np
import kaggle

################################################
    
def getProbDistQuestionA(JLst, a):
    '''
    Implement a function to calculate P (J|α) with input arguments J, α and output P (J|α)
    JLst: a sequence of jar
    a is the model paameter deciding the next jar same as the previous one or not
    '''
 
    PJGivenA = 1          #prob of jar sequence given  \alpha
    for i in range(1, len(JLst)):
        PJGivenA = PJGivenA * a if JLst[i] == JLst[i-1] else PJGivenA * (1-a)
                   
    print ("PJGivenA: ", PJGivenA)

    return PJGivenA

def getProbDistQuestionB(JLst, BLst):
    '''
    Implement a function to calculate P (BjJ) with input arguments J, B and output P (BjJ).
    ''' 
    PBGivenJ = 1               # prob of Ball sequences given Jar sequences
    for i in range(0, len(JLst)):
        if JLst[i] == 0:
            PBGivenJ = PBGivenJ * 0.2 if BLst[i]== 0 else PBGivenJ * 0.8
        elif JLst[i] == 1:
            PBGivenJ = PBGivenJ * 0.9 if BLst[i]== 0 else PBGivenJ * 0.1
                
    print ("PBGivenJ: ", PBGivenJ)

    return PBGivenJ

def getProbDistQuestionC(a):
    '''
    Implement a function to calculate P(α) with input argument α and output P(α
    get prior probability p(a)
    '''
    PA  =0       #pa, probability of \alpha
    if a >= 0 and a <= 1:
        PA = 1
        
    print ("PA: ", PA)

    return PA


    
def getProbDistQuestionD(JLst, BLst, a):
    '''
    Implement a function to calculate P(α; J; B) with input arguments J, B, α and output
    P(α; J; B).
    '''
    PJGivenA = getProbDistQuestionA(JLst, a)
    PBGivenJ = getProbDistQuestionB(JLst, BLst)
    PA = getProbDistQuestionC(a)
    
    PJointAJB = PJGivenA * PBGivenJ * PA           #joint probablity of P(α, J, B).
    print ("PJointAJB: ", PJointAJB)



if __name__== "__main__":

    ################################################
    print('1a through 1l computation goes here ...')
    
    '''
    print ("beginning Question 1a")

    JLst = [0, 1, 1, 0, 1]
    a = 0.75
    getProbDistQuestionA(JLst, a)
    
    JLst = [0, 0, 1, 0, 1]
    a = 0.2
    getProbDistQuestionA(JLst, a)
    
    JLst = [1,1,0,1,0,1]
    a = 0.2
    getProbDistQuestionA(JLst, a)
    
    JLst = [0,1,0,1,0,0]
    a = 0.2
    getProbDistQuestionA(JLst, a)
    
    print ("beginning Question 1b")
    JLst = [0,1,1,0,1]
    BLst = [1,0,0,1,1]
    getProbDistQuestionB(JLst, BLst)
    
    JLst = [0,1,0,0,1]
    BLst = [0,0,1,0,1]
    getProbDistQuestionB(JLst, BLst)
    
    JLst = [0,1,1,0,0,1]
    BLst = [1,0,1,1,1,0]
    getProbDistQuestionB(JLst, BLst)
    
    JLst = [1,1,0,0,1,1]
    BLst = [0,1,1,0,1,1]
    getProbDistQuestionB(JLst, BLst)
    
    
    print ("beginning Question 1c")
    getProbDistQuestionC(2)
    
    '''
    
    print ("beginning Question 1d")
    
    JLst = [0,1,1,0,1]
    BLst = [1,0,0,1,1]
    a = 0.75
    getProbDistQuestionD(JLst, BLst, a)
    
    JLst = [0,1,0,0,1]
    BLst = [0,0,1,0,1]
    a = 0.3
    getProbDistQuestionD(JLst, BLst, a)
    
    JLst = [0,0,0,0,0,1]
    BLst = [0,1,1,1,0,1]
    a = 0.63
    getProbDistQuestionD(JLst, BLst, a)
    
    JLst = [0,0,1,0,0,1,1]
    BLst = [1,1,0,0,1,1,1]
    a = 0.23
    getProbDistQuestionD(JLst, BLst, a)
    


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