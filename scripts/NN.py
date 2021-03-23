import numpy as np
import random
import math
import sklearn.utils


class NeuralNetwork:
    def __init__(self,x,y,setup,activation,batchSize,is_autoencoder=True,lr=.05,seed=1,iters=500):
        #Note - these paramaters are examples, not the required init function parameters

        np.random.seed(seed)

        self.x = x
        self.y = y

        # hyperparameters
        self.lr = lr
        self.setup = setup
        self.iters = iters
        self.activation = activation
        self.batchSize = batchSize
        self.is_autoencoder = is_autoencoder

        self.hidden_weights = np.random.uniform(size=(self.setup[0][0], self.setup[0][1])) # 8, 3
        self.output_weights = np.random.uniform(size=(self.setup[1][0], self.setup[1][1])) # 3, 8

        self.hidden_biases = np.random.uniform(size=(1, self.setup[0][1])) # 3
        self.output_biases = np.random.uniform(size=(1, self.setup[1][1])) # 8

        self.input_act = np.random.uniform(size=(1, self.setup[0][0])) # 8
        self.hidden_act = np.random.uniform(size=(1, self.setup[0][1])) # 3
        self.output_act = np.random.uniform(size=(1, self.setup[1][1])) # 8


    def feedforward(self, input_act):

        self.input_act = input_act

        hidden_z = np.dot(input_act, self.hidden_weights) + self.hidden_biases

        hidden_act = activation(hidden_z, self.activation)

        output_z = np.dot(hidden_act, self.output_weights) + self.output_biases

        output_act = activation(output_z, self.activation)

        self.hidden_act = hidden_act
        self.output_act = output_act



    def backprop(self, target):
        
        error = target - self.output_act # div by batchsize if doing BGD

        o_gradient = activation_derivative(self.output_act, self.activation)

        h_gradient = activation_derivative(self.hidden_act, self.activation)

        delta_out = error * o_gradient # should the gradients be negative here?

        #print(delta_out)

        delta_hidden = np.dot(delta_out, self.output_weights.T) * h_gradient # changed dot prod here from self.hidden_weights to self.output_weights.T

        self.output_weights += np.dot(self.hidden_act.T, delta_out) * self.lr

        self.hidden_weights += np.dot(self.input_act.T, delta_hidden) * self.lr

        self.output_biases += np.sum(delta_out, axis=0, keepdims=True) * self.lr

        self.hidden_biases += np.sum(delta_hidden, axis=0, keepdims=True) * self.lr


    def fit(self, verbose=False):

        loss = 999
        loss_iter_sample = []

        for i in range(self.iters):
            i+=1

            shuffled = sklearn.utils.shuffle(self.x, self.y) # does not shuffle in place
            shuffled_x, shuffled_y = shuffled[0], shuffled[1]

            for s in range(self.batchSize): #range(len(shuffled_x)):

                sample = shuffled_x[s]
                target = shuffled_y[s]

                #print(sample)

                sample = np.reshape(sample, (1,self.setup[0][0]))
                target = np.reshape(target, (1,self.setup[1][1]))

                self.feedforward(sample)
                self.backprop(target)

                prev_loss = loss
                loss = np.sum((target - self.output_act)**2)

                # if math.isnan(loss):
                #     loss = prev_loss
                #     break

                loss_iter_sample.append((loss, i, s))

                if verbose:
                    print('loss = %.2f. iter = %d. sample = %d' % (loss, i, s))

        return loss_iter_sample



        
    def predict(self, input_act):
        
        self.feedforward(input_act)

        if self.is_autoencoder:

            i = np.argmax(self.output_act)
            pred = np.zeros(shape=self.output_act.shape)
            pred[0][i] = 1

            return pred.flatten()

        else:

            return self.output_act







def activation(x, activ):
    """
    Takes a numpy array and returns a numpy array of the same shape after applying the specified activation function.
    """
    if activ == 'sigmoid':
        return 1/(1+np.exp(-x))
    elif activ == 'relu':
        return np.maximum(0, x)
    else:
        raise Exception('Activation function "%s" is not supported' % activ)



def activation_derivative(x, activ):
    if activ == 'sigmoid':
        # derivative of the sigmoid function is (sig(x) * (1 - sig(x)))
        return activation(x, activ) * (1 - activation(x, activ))
    elif activ == 'relu':
        x_copy = np.array(x, copy=True)
        x_copy[x<=0] = 0
        x_copy[x>0] = 1
        return x_copy
    else:
        raise Exception('Activation function "%s" is not supported' % activ)



def one_hot_encode_dna(seq):
    one_hot = []
    for nt in seq.upper():
        if nt == 'A':
            one_hot.append([1,0,0,0])
        elif nt == 'T':
            one_hot.append([0,1,0,0])
        elif nt == 'C':
            one_hot.append([0,0,1,0])
        elif nt == 'G':
            one_hot.append([0,0,0,1])
    flat_arr = np.array(one_hot).flatten() # convenient way of combining a list of lists
    return np.array(flat_arr)


def make_training_examples(pos_file, neg_file):

    training_examples = {}

    # fill dictionary with positive examples
    with open(pos_file, 'r') as f:
        for line in f:
            training_examples[line.rstrip()] = 1

    # add 3 negative examples for each positive example (137 = 3 = 411)
    # first, count the lines in the negative sequences file and choose random indeces
    # corresponding to lines in the file that will be sampled for negative sequences
    with open(neg_file, 'r') as f:
        numLines = 0
        for line in f:
            if line[0] == '>':
                continue
            else:
                numLines+=1

        # tried to get fancy with choosing random lines, but I didn't want to deal with duplicates
        #neg_indeces = random.choices(list(range(0,numLines)), k=411)

        # now, the choosing of negative examples is deterministic
        neg_indeces = list(range(0,numLines,round(numLines/411)))

    # make the first 17 characters from the line a negative example, and add it to
    # the training_examples dictionary
    with open(neg_file, 'r') as f:

        # random neg example, in the extremely rare case that we need to use prev_line for the first negative index
        prev_line = 'TTTAGTTGGTTCTTTTT'

        i = 0
        for line in f:
            if line[0] == '>':
                continue
            else:
                if i in neg_indeces:
                    if line[:17].rstrip() in training_examples:
                        training_examples[prev_line[:17].rstrip()] = 0
                    else:
                        training_examples[line[:17].rstrip()] = 0
                prev_line = line
                i+=1

    return training_examples


def make_autoencoder_training_examples():
    unique_bvs = []

    for i in range(8):
        ex = []
        for j in range(8):
            if i == j:
                ex.append(1)
            else:
                ex.append(0)
        unique_bvs.append(np.array(ex))

    training_examples = []
    # the classifier has 548 examples, so I'll give the autoencoder 560
    # for i in range(70):
    #     training_examples = training_examples + unique_bvs

    training_examples = training_examples + unique_bvs

    training_examples = np.array(training_examples)
    #training_examples = np.reshape(training_examples, (560, 8))

    return unique_bvs # training_examples









###
# UNIT TESTS
###
seq = 'actgactg'
def one_hot_encode_test(seq):
    one_hot = one_hot_encode_dna(seq)
    print(one_hot)

#one_hot_encode_test(seq)




