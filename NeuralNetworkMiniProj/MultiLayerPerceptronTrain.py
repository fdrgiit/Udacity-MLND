#
#   In the following exercises we will complete several functions for a simple 
#   implementation of neural networks based on code by Roland Szabo.
#
#   In this exercise, we will begin writing a function, fit(), which will train our 
#   network on data that we provide.
#
#   Special thanks to Roland Szabo for the use of his code as a basis for this and 
#   preceding exercises. His original code can be found at 
#   http://rolisz.ro/2013/04/18/neural-networks-in-python/
#
#   NOTE: this and preceding exercises creating classes for functioning
#   neural networks are HARD, and are not efficient implementations.
#   Consider them an extra challenge, not a requirement!



import numpy as np

#choose a seed for testing in the exercise
np.random.seed(1)

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

    def __init__(self,weights = [0]):
        if type(weights) in [type([]),type(np.array([]))]:
            self.weights = weights


class NeuralNetwork:

    def __init__(self, layers):
        """
        :param layers:  A list containing the number of units in each layer. Should be 
                        at least two values
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
        
#        for x in self.nodes:
#            print len(x),"-nodes Layer has weigts # of", len(x[-1].weights)


    def predict(self, x):
        """
        :param x: a 1D ndarray of input values
        :return: a 1D ndarray of values of output nodes
        """
        a=np.ones(x.shape[0]+1)
        a[0:-1]=x    # first layer takes one more element than input vector to account for bias
        for l in range(1, len(self.nodes)):
            a = [node.activate(a) for node in self.nodes[l]]
#            print a
        return a
    
    def fit(self, X, y, learning_rate=0.2, epochs=3000):
        self.BackPropagation(X, y, learning_rate, epochs)
    
    def BackPropagation(self, X, y, learning_rate=0.2, epochs=3000):
        """
        :param X: a 2D ndarray of many input values
        :param y: a 2D ndarray of corresponding desired output vectors
        :param learning_rate: controls the learning rate (optional)
        :param epochs: controls the number of training iterations (optional)
        """

        #YOUR CODE HERE
        for epoch in range(epochs):
        #In each epoch, we will choose and train on an example from X, y
            sel = np.random.randint(X.shape[0]) # select a input vector to train
            vout = self.predict(X[sel])
        #to train on each example, we will first need to evaluate the example from X
        #storing the signal strength at each node before the activation is applied.

        #Then compare the outputs in y to our outputs, and scale them by the 
        #activation_derivative(strength) at the signal strengths for each of the output
        #nodes.
            self.deltas(y[sel], vout, -1)
        #Iterate backwards over the layers, using the deltas method below to associate a
        #rate of change to each node
            for l in range(len(self.nodes) - 2, 0, -1):
                output_layer = [j.last_input for j in self.nodes[l]]
                self.deltas(y[sel], output_layer, l)
        #then modify each of the (non-input) node's weights by the learning rate times  
        #the current node's delta times the previous node's last input.
            prevs = np.ones(X[sel].shape[0] + 1)
            prevs[0:-1] = X[sel]    # first layer takes one more element than input vector to account for bias
            for l in range(1, len(self.nodes)):
                for node in self.nodes[l]:
                    node.weights += learning_rate * node.delta * prevs
                prevs = np.array([prev.last_input for prev in self.nodes[l]])


    def deltas(self,y,outputs,layer):
        '''
        :param y: an array of expected outputs
        :param ouptuts: an array of actual outputs from the layer
        :param layer: which layer of the network to update. Use -1 for output layer.
        sets the delta values for the units in the layer
        :returns null:
        '''

        if layer==-1:
            final = [y[i]-outputs[i] for i in range(0,len(y))]
        else:
            final = []
            for i in range(0,len(self.nodes[layer])):
                sum=0
                for j in range(0,len(self.nodes[layer+1])):
                    sum+= self.nodes[layer+1][j].weights[i] * self.nodes[layer+1][j].delta
                final.append(sum)
        for i in range(0,len(self.nodes[layer])):
            self.nodes[layer][i].delta = logistic_derivative(outputs[i])*final[i]


