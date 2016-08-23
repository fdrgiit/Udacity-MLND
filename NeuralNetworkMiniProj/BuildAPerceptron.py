#-----------------------------------

#
#   In this exercise you will put the finishing touches on a perceptron class
#
#   Finish writing the activate() method by using numpy.dot and adding in the thresholded
#   activation function

import numpy

class Perceptron:

    
    def activate(self,inputs):
        '''Takes in @param inputs, a list of numbers.
        @return the output of a threshold perceptron with
        given weights, threshold, and inputs.
        ''' 
               
        
        #YOUR CODE HERE

        #TODO: calculate the strength with which the perceptron fires
        a = numpy.dot(self.weights, inputs)
        
        #TODO: return 0 or 1 based on the threshold
        if a >= self.threshold:
            result = 1
        else:
            result = 0
        return result

        
        
    def __init__(self,weights=None,threshold=None):
        if weights:
            self.weights = weights
        self.threshold = 0
        if threshold:
            self.threshold = threshold
            
            