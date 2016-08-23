#-----------------------------------

#
#   In this exercise we write a perceptron class
#   which can update its weights
#
#   Your job is to finish the train method so that it implements the perceptron update rule

import numpy as np

class Perceptron:
    
    def activate(self,values):
        '''Takes in @param values, @param weights lists of numbers
        and @param threshold a single number.
        @return the output of a threshold perceptron with
        given weights and threshold, given values as inputs.
        ''' 
               
        #First calculate the strength with which the perceptron fires
        strength = np.dot(values,self.weights)
        print strength, " in activate"
        #print self.threshold, " self threshold"
        result = 0
        if strength > self.threshold:
            result = 1
        #print result, " in activate"    
        return result

    def updateOnce(self,values,train,eta=.1):
        '''Takes in a 2D array @param values consisting of a LIST of inputs
        and a 1D array @param train, consisting of a corresponding list of 
        expected outputs.
        Updates internal weights according to the perceptron training rule
        using these values and an optional learning rate, @param eta.
        '''
        #YOUR CODE HERE
        #update self.weights based on the training data
        #print values
        outputs = [self.activate(i) for i in values]
        outs = np.array(outputs)       
        print "outputs:", outs, "____trains: ", train
        deltaW = eta * (np.array(train) - outs)
        for i in range(len(train)) :
            self.weights = deltaW[i] + self.weights
    
    def update(self,values,train,eta=.1):
        # Instead of checking no error anymore I let the update go 20 times, kind of cheating
        for k in range(20) :
            self.updateOnce(values, train, eta)
            print "weights: ", self.weights, "k = ", k
        
    def __init__(self,weights=0,threshold=0):
            self.weights = weights
            self.threshold = threshold