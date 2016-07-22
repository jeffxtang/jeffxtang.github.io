import numpy as np
from scipy.spatial import distance
import sys

# eat|apple,eat|orange,eat|rice,drink|juice,drink|milk,drink|water,orange|juice,apple|juice,rice|milk,milk|drink,water|drink,juice|drink
# the|dog,dog|saw,saw|a,a|cat,the|dog, dog|chased,chased|the,the|cat,the|cat,cat|climbed,climbed|a,a|tree
#sentences = ["eat apple","eat orange","eat rice","drink juice","drink milk","drink water","orange juice","apple juice","rice milk","milk drink","water drink","juice drink"]
sentences = ['the dog saw a cat', 'the dog chased the cat', 'the cat climbed a tree']

# get vocabulary list from sentences
vocabulary = []
for words in [sentence.split() for sentence in sentences]:
    for word in words:
        if not word in vocabulary:
            vocabulary += [word]

# to get a word's index, use:
# vocabulary.index(word) # same as vocabulary[word] in word2vec-basic.py

# each word in the input layer is represented as a one-hot encoded vector:
def word_one_hot(word):
    one_hot = [0] * len(vocabulary)
    one_hot[vocabulary.index(word)] = 1
    return one_hot

# word_one_hot('cat') returns [0, 0, 0, 0, 1, 0, 0, 0]

learning_rate = 0.2
V, N = len(vocabulary), 5
WI = (np.random.random((V, N)) - 0.5) / N
WO = (np.random.random((N, V)) - 0.5) / V


# the function takes 4 parameters:
# X is the input one-hot vector for a context word (aka input word)
# weights2h is the weights matrix from the input layer to the hidden layer
# weights2o is the weights matrix from the hidden layer to the output layer
# returns a tuple of the output vector in the output layer, calculated from X, weightsI2H and weightsH2O) and
# the output value in the hidden layer (used in backpropagation)
# NOTE: Word2vec uses only one hidden layer, so 2 specific weights parameters are provided here.
# But more generally, a NN can be represented as an input vector and a list of weight matrices.
def feedforward(X, weights2h, weights2o):
    hidden_netinputs = np.dot(X, weights2h) # no need to convert the X vector (a Python list, returned from calling word_one_hot) to np.array
    hidden_outputs = hidden_netinputs # linear activation funtion used for the hidden layer neuron
    output_netinputs = np.dot(hidden_outputs, weights2o)

    numerator = np.array([sys.float_info.max if x == float('inf') else x for x in np.exp(output_netinputs)])
    denominator = np.sum([sys.float_info.max if x == float('inf') else x for x in np.exp(output_netinputs)])
    output_outputs = numerator / denominator

    #output_outputs = np.exp(output_netinputs) / np.sum(np.exp(output_netinputs))

    return hidden_outputs, output_outputs

# backpropagation function to update weights, with the folloring parameters:
# output_hidden, output_output are the output vectors in the hidden layer (size Nx1) and output layer (size Vx1)
# target is the expected word, also a one-hot vector
def backpropagation(input_one_hot, target_one_hot, hidden_outputs, output_outputs):
    global WO
    global WI
    # 1. update the weights between the hidden and output layers
    # first define our error function (aka loss function) E
    #error = - np.dot(target_one_hot, np.log(output_outputs))

    # then calculate the derivative of E wrt the output, net input, and weight (hidden to output) of the output layer
    # TODO: which one? - should be the second one because for different neuron's (in the output layer) output, error is different (all 0's except the neuron with target value 1)
    #derror_doutput_outputs = -1.0 / output_outputs
    #derror_doutput_outputs = -1.0 * (target_one_hot / output_outputs)

    # TODO: figure out which one of the following two is correct? should be the first one, based on calculus.
    #doutput_outputs_doutput_netinputs = output_outputs * (1 - output_outputs) # derivative of softmax (output_output = np.exp(netinput_output) / sum(np.exp(netinput_output)))
    # doutput_outputs_doutput_netinputs = output_outputs * (target_one_hot - output_outputs) # derivative of softmax (output_output = np.exp(netinput_output) / sum(np.exp(netinput_output)))

    derror_doutput_netinputs = output_outputs - target_one_hot
    doutput_netinputs_dweights2o = hidden_outputs
    #derror_dweights2o = np.array([derror_doutput_outputs * doutput_outputs_doutput_netinputs * x for x in doutput_netinputs_dweights2o])
    derror_dweights2o = np.array([derror_doutput_netinputs * x for x in doutput_netinputs_dweights2o])

# In [1082]: a=np.array([2,3])
#
# In [1083]: b=np.array([1, 10, 100])
#
# In [1084]: np.array([x * b for x in a])
# Out[1084]:
# array([[  2,  20, 200],
#        [  3,  30, 300]])

    # in vector_math.js:
    # error_i = n['value'] - expectedOutput[i]
    # e['gradient'] = errors[i] * hiddenNeurons[j]['value'];
    # e['weight'] -= learning_rate * e['gradient'];

    # apply weight update
    WO = WO - learning_rate * derror_dweights2o

    # 2. update the weights between the input and hidden layers
    # calculate the derivative of E wrt the output, net input, and weight (input to hidden) of the hidden layer
    doutput_netinputs_dhidden_outputs = WO

    # TODO: which one? STILL NOT 100% certain
    #derror_dhidden_outputs = derror_doutput_outputs * doutput_outputs_doutput_netinputs * doutput_netinputs_dhidden_outputs
    # which is target_one_hot * (output_outputs - 1) = target_one_hot * output_outputs - target_one_hot
    #derror_doutput_netinputs = output_outputs - target_one_hot

    #derror_dhidden_outputs = derror_doutput_netinputs * doutput_netinputs_dhidden_outputs
    # the following is a big change: sigma (sum) is used (backpropagation step by step: "the output of each hidden layer
    # neuron contributes to the output (and therefore error) of multiple output neurons. ")
    #derror_dhidden_outputs = [np.dot(x, derror_doutput_netinputs) for x in doutput_netinputs_dhidden_outputs] # list of N values
    derror_dhidden_outputs = np.dot(doutput_netinputs_dhidden_outputs, derror_doutput_netinputs)

    dhidden_outputs_dhidden_netinputs = 1 # because linear activation function is used
    dhidden_netinputs_dweights2h = input_one_hot # list of V values

    # derror_dweights2h = derror_dhidden_outputs * dhidden_outputs_dhidden_netinputs * dhidden_netinputs_dweights2h
    derror_dweights2h = np.dot(np.array(dhidden_netinputs_dweights2h).reshape(V, 1), np.array(derror_dhidden_outputs).reshape(1,N))

    # apply weight update
    WI = WI - learning_rate * derror_dweights2h

def training():
    for sentence in sentences:
        first = sentence.split()[:-1] # for sentence 'the cat climbed a tree', this is ['the', 'cat', 'climbed', 'a']
        second = sentence.split()[1:] # ['cat', 'climbed', 'a', 'tree']
        for (context_word, target_word) in zip(first, second): # zip(first, second) is [('the', 'cat'), ('cat', 'climbed'), ('climbed', 'a'), ('a', 'tree')]
            #print context_word, target_word
            hidden_outputs, output_outputs = feedforward(word_one_hot(context_word), WI, WO)
            backpropagation(word_one_hot(context_word), word_one_hot(target_word), hidden_outputs, output_outputs)


word_test = "chased"
for w in vocabulary:
    print word_test, ":", w, "=", distance.euclidean(WI[vocabulary.index(word_test)], WI[vocabulary.index(w)])
