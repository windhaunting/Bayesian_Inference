# Import python modules
import numpy as np
import kaggle

################################################
    
def testQuestionAProbDist(JLst, a):
    '''
    Implement a function to calculate P (J|α) with input arguments J, α and output P (J|α)
    JLst: a sequence of jar
    a is the model paameter deciding the next jar same as the previous one or not
    '''
 
    PJGivenA = 1          #prob of jar sequence given  \alpha
    for i in range(1, len(JLst)):
        PJGivenA = PJGivenA * a if JLst[i] == JLst[i-1] else PJGivenA * (1-a)
                   
    print ("PJGivenA: ", PJGivenA)


    
if __name__== "__main__":

    ################################################
    print('1a through 1l computation goes here ...')
    JLst = [0, 1, 1, 0, 1]
    a = 0.75
    
    testQuestionAProbDist(JLst, a)
    
    JLst = [0, 0, 1, 0, 1]
    a = 0.2
    testQuestionAProbDist(JLst, a)
    
    
    JLst = [1,1,0,1,0,1]
    a = 0.2
    testQuestionAProbDist(JLst, a)
    
    JLst = [0,1,0,1,0,0]
    a = 0.2
    testQuestionAProbDist(JLst, a)
    
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