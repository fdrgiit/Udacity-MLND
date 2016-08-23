#
#   In the following exercises we will complete several functions for 
#   a simple implementation of neural networks based on code by Roland 
#   Szabo. 
#
#   In this exercise, we will begin by writing a function, deltas(), 
#   which will compute and store delta factors for each node in a 
#   layer, given the deltas for the previous layer.
#
#   Recall that the delta value associated to an output node is the 
#   activation_derivative 
#   of the node's last_input multiplied by the difference of its expected output minus 
#   its actual output
#
#   The delta value associated to a hidden node is the activation_derivative of the 
#   node's last_input times the sum over the next layer of the products of each nodes
#   delta value times weight from the current node
#
#   NOTE: the following exercises creating classes for functioning
#   neural networks are HARD, and are not efficient implementations.
#   Consider them an extra challenge, not a requirement!



import numpy as np


def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class Sigmoid:

    # keeps track of previous input strengths for backpropagation
    last_input = 0
    # space to keep track of deltas for backpropagation
    delta = 0

    def activate(self,values):
        '''Takes in @param values, @param weights lists of numbers
        and @param threshold a single number.
        @return the output of a threshold perceptron with
        given weights and threshold, given values as inputs.
        '''

        #First calculate the strength with which the perceptron fires
        strength = self.strength(values)
        self.last_input = strength

        result = logistic(strength)

        return result

    def strength(self,values):
        # Formats inputs to easily compute a dot product
        local = np.atleast_2d(self.weights)
        values = np.transpose(np.atleast_2d(values))
        strength = np.dot(local,values)
        return float(strength)

    def __init__(self,weights=None):
        if type(weights) in [type([]), type(np.array([]))]:
            self.weights = weights




class NeuralNetwork:

    def __init__(self, layers):
        """
        :param layers: A list containing the number of units in each layer. Should be at least two values
        """

        self.nodes = [[]]
        #input nodes
        for j in range(0,layers[0]):
            self.nodes[0].append(Sigmoid())
        #randomly initialize weights
        for i in range(1, len(layers)-1):
            self.nodes.append([])
            for j in range(0,layers[i]+1):
                self.nodes[-1].append(Sigmoid((2*np.random.random(layers[i - 1]+1)-1)*.25))
        self.nodes.append([])
        for j in range(0,layers[i+1]):
            self.nodes[-1].append(Sigmoid((2*np.random.random(layers[i]+1)-1)*.25))
        print "nodes #. in each layer: ", layers


    def predict(self, x):
        """
        :param x: a 1D ndarray of input values
        :return: a 1D ndarray of values of output nodes
        """
        print x
        x = np.array(x)
        print x, x.shape[0]
        a=np.ones(x.shape[0]+1)
        print a
        a[0:-1]=x
        print a
        for l in range(1, len(self.nodes)):
            print "iter: ", l
            a = [node.activate(a) for node in self.nodes[l]]
        return a

    def deltas(self,y,outputs,layer):
        '''
        :param y: an array of expected outputs (in the case of an output layer) or deltas from the previous layer (in the case of an input layer)
        :param ouptuts: an array of actual outputs from the layer
        :param layer: which layer of the network to update.
        sets the delta values for the units in the layer
        :returns: a list of the delta values for use in the next previous layer
        '''

        #YOUR CODE HERE
        if layer==-1:
            final = [y[i]-outputs[i] for i in range(0,len(y))]
        else:
            final = []
            for i in range(0,len(self.nodes[layer])):
                sum=0
                for j in range(0,len(self.nodes[layer+1])):
                    sum+= self.nodes[layer+1][j].weights[i] * self.nodes[layer+1][j].delta
                final.append(sum)
        
        res = []
        for i in range(0,len(self.nodes[layer])):
            self.nodes[layer][i].delta = logistic_derivative(outputs[i])*final[i]
            res.append(self.nodes[layer][i].delta)
        return res