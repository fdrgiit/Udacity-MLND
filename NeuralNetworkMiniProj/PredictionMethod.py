#
#   In the following exercises we will complete several functions for a 
#   simple implementation of neural networks based on code by Roland 
#   Szabo.
#   
#   In this exercise, we will will write a function, predict(),
#   which will predict the value of given inputs based on a constructed 
#   network.
#
#   Note that we are not using the Sigmoid class we implemented earlier
#   to be able to compute more efficiently.
#
#   NOTE: the following exercises creating classes for functioning
#   neural networks are HARD, and are not efficient implementations.
#   Consider them an extra challenge, not a requirement!



import numpy as np

#choose a seed for testing in the exercise
#np.random.seed(1)

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))
    
class NeuralNetwork:

    def __init__(self, layers):
        """
        :param layers: A list containing the number of units in each
          layer. Should be at least two values
        """
        self.activation = logistic
        self.activation_deriv = logistic_derivative

        print "layers: ", layers
        self.weights = []
        #randomly initialize weights)
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)
            print "i and weights = ", i, self.weights
        self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)
        print "i and weights = ", i, self.weights

    def predict(self, x):
        """
        :param x: a 1D ndarray of input values
        :return: a 1D ndarray of values of output nodes
        """
        
        #YOUR CODE HERE
        x.append(1)
        print "intput:", x
        print "self.weights: ", self.weights
        #our neural network is a numpy array self.weights
        #its first dimension is layers; self.weights[0] is the first
        #(input) layer.
        #its second dimension is nodes; self.weights[1][3] is the 4th 
        #node in the second (hidden) layer.
        #its third dimension is weights; self.weights[1][3][2] will be 
        #the weight assigned to the input from the third node on the 
        #first layer.
        
        #for each layer, evaluate the nodes in that layer
        #by taking the dot product of the output of the previous layer
        #(or the input in the case of the first layer)
        #with the weights for that node, then applying the activation
        #function, self.activation()
        cur = x;
        for layer in range(len(self.weights)):
            nexts = np.dot(cur, self.weights[layer])
            print nexts
            cur = [self.activation(i) for i in nexts]
            print cur
        #also make sure to add a constant dummy value to the input by 
        #appending 1 to it
        
        #return the output vector from the last layer.
        return cur[0]
