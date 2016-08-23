#
#   In this exercise, you will create a network of perceptrons which
#   represent the xor function use the same network structure you used
#   in the previous quizzes.
#
#   You will need to do two things:
#   First, create a network of perceptrons with the correct weights
#   Second, define a procedure EvalNet() which takes in a list of 
#   inputs and ouputs the value of this network.

import numpy as np

class Perceptron:

    def evaluate(self,values):
        '''Takes in @param values, @param weights lists of numbers
        and @param threshold a single number.
        @return the output of a threshold perceptron with
        given weights and threshold, given values as inputs.
        ''' 
               
        #First calculate the strength with which the perceptron fires
        strength = np.dot(values,self.weights)
        
        #Then evaluate the return value of the perceptron
        if strength >= self.threshold:
            result = 1
        else:
            result = 0

        return result

    def __init__(self,weights=0,threshold=0):
        if weights:
            self.weights = weights
        if threshold:
            self.threshold = threshold
            

Network = [
    #input layer, declare perceptrons here
    [ Perceptron([1, 0], 0.5), Perceptron([0.5, 0.5], 0.75), Perceptron([0, 1], 0.5)], \
    #output node, declare one perceptron here
    [ Perceptron([1, -2, 1], 1) ]
]


def EvalNetwork(inputValues, Network):
    cur = inputValues
    for i in range(len(Network)):
        nexts = [j.evaluate(cur) for j in Network[i]]
        cur = nexts
    OutputValues = cur[0]
    # Be sure your output values are single numbers
    return OutputValues