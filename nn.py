# Backprop implementation is not vectorized.
import time
import numpy as np
import math
import matplotlib.pyplot as plot


def sigmoid(x):
    ''' Returns the logistic function of the input '''
    #probably kill this and put it in feed_forward()
    return 1 / (1 + np.exp(-x))


def node_output(weights, inputs):
    ''' Takes a vector of weights and a vector of inputs
    and computes the dot product of the two vectors, and 
    performs the activation function on the result '''
    return sigmoid(np.dot(weights, inputs))


def feed_forward(network, inputs):
    ''' Takes a network and an input vector and propagates the input
    through the network. The return value is a list of the 
    outputs of each neuron in the network, the last being 
    the classifier's hypothesis. '''

    all_outputs = []

    # make the input dictionary into list
    inputs = list(inputs.values())

    # remove the training label
    del inputs[-1]

    # turn the list into a numpy array
    inputs = np.asarray(inputs)
    
    for layer in network:

        # add a bias input 
        inputs_wb = np.insert(inputs, 0, 1)

        # compute the outputs of the neuron in this layer
        output = sigmoid(layer.dot(inputs_wb))

        # add this to the list of ouputs
        all_outputs.append(output)

        # feed the ouputs into the inputs of the next layer
        inputs = output

    return all_outputs


def feed_forward_depreciated(network, inputs):
    ''' Unvectorised version of the preceeding function Resulted in
    overhead of up to 50%. To be deleted. '''
    all_outputs = []

    # make the input dictionary into list
    inputs = list(inputs.values())

    # get rid of the training label 
    #(not so nice)
    del inputs[-1]

    # for each layer in the network
    for layer in network:
        # insert the bias at the front of the input vector
        inputs_wb = np.insert(inputs, 0, 1)

        # compute the output vector for the neurons in this layer
        output = np.array([node_output(neuron, inputs_wb) 
                 for neuron in layer])

        # add to the list of outputs for the whole network
        all_outputs.append(output) 
        
        # make the output the input for the next layer
        inputs = output

    return all_outputs


def backprop(network, input_vec):
    ''' Gets the result of forward propagating some input vector 
    through some network, and then performs backpropagation of the 
    result's error through the network. '''
    # todo: generalise for multilayer networks

    # Try chaning this!
    learning_rate = 0.2

    # get the label for this input vector
    target = input_vec['y']
    
    # propagate the input_vec through the network
    all_outputs = feed_forward(network, input_vec)

    # convert the input vector to a list
    input_vec = list(input_vec.values())

    # remove the training label value
    #(can fix by having a separate list of the labels)
    del input_vec[-1]

    # get the final result of the network, which is the 
    # prediction of the classifier
    output = all_outputs[-1][0]

    # get the outputs of the previous layer
    hidden_outputs = all_outputs[0]

    # error in the result where (1 - output) is from the derivative 
    # of the sigmoid function
    output_delta = output * (1 - output) * (output - target)

    # get the weights of the output neuron
    output_neuron = network[-1][0]

    # for each node in the hidden layer
    for i, hidden_output in enumerate(np.insert(hidden_outputs, 0, 1)):

        # Adjust the weight going into the output from that node
        output_neuron[i] -= learning_rate * output_delta * hidden_output

    # get errors for hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) * 
                     (output_delta * output_neuron[i]) 
                     for i, hidden_output in enumerate(hidden_outputs)]

    # adjust weights for neurons in the hidden layer
    for i, hidden_neuron in enumerate(network[0]):

        # for each neuron, adjust the j'th weight
        for j, input_ in enumerate([1] + input_vec):

            hidden_neuron[j] -= learning_rate * hidden_deltas[i] * input_

    return network


def initialize_network(layers):
    ''' Returns a network with layers[i] nodes in the i'th
    layer with random intitialzed values in the range [-a,a].
    '''

    # set the range of the inital weights.
    a = 1

    # initialise a network as specified
    network = [(np.random.rand(layers[n], layers[n-1] + 1)*2*a - a)
                    for n in range(1, len(layers))]

    return network


def train(network, training_set, testing_set, n, a=False):
    ''' Takes a network and a set of training data, and performs
    n interations of backpropagation on the training data set. An 
    optional argument 'a' can be made True to test the networks 
    accuracy at the beginning of each epoch. This creates overhead.
    TODO: instead interate on the data until the network converges.'''

    if a:
        accuracy_time = []

        for i in range(n):
            # asses the network's accuracy
            accuracy = network_accuracy(network, testing_set)

            # record the accuracy
            accuracy_time.append(accuracy)

            print("--- Dawn of epoch " + str(i + 1) + " of " + str(n) + ". " +
                  "Accuracy is " + str(accuracy) + "%.")

            # for each event in the training set, perform backpropagation
            for event in training_set:
                backprop(network, event)

        plot.plot(accuracy_time)
        plot.show()

    else:
        for i in range(n):
            print("--- Dawn of epoch " + str(i + 1) + " of " + str(n) + ". ")

            # for each event in the training set, perform backpropagation
            for event in training_set:
                backprop(network, event)

    return network


def network_accuracy(network, testing_set):
    ''' A function which evaluates the accuracy of a network
    given a testing set. This function allows tracking of accuracy
    through training epochs, but is optional as it creates extra
    overhead. The function returns the network accuracy as a percent. '''

    num_correct = 0

    # for 100 test events
    for event in testing_set[:100]:

        # get the result of forward propagation
        result = feed_forward(network, event)[-1][0]

        # check if the result is correct
        if event['y'] == 1 and result > 0.5:
            num_correct += 1
        elif event['y'] == 0 and result < 0.5:
            num_correct += 1 

    return num_correct
























