import numpy as np
import random
import math
import sklearn.utils


class NeuralNetwork:
    def __init__(self,x,y,setup,activation,batchSize,is_autoencoder=True,lr=.05,seed=1,iters=500):

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

        self.hidden_weights = np.random.uniform(size=(self.setup[0][0], self.setup[0][1]))
        self.output_weights = np.random.uniform(size=(self.setup[1][0], self.setup[1][1]))

        self.hidden_biases = np.random.uniform(size=(1, self.setup[0][1]))
        self.output_biases = np.random.uniform(size=(1, self.setup[1][1]))

        self.input_act = np.random.uniform(size=(1, self.setup[0][0]))
        self.hidden_act = np.random.uniform(size=(1, self.setup[0][1]))
        self.output_act = np.random.uniform(size=(1, self.setup[1][1]))


    def feedforward(self, input_act):
        """
        Calculates the activations of the nodes in the hidden layer given the activations of the input layer,
        the current weights of the edges between the input and hidden layers, and the current biases of the hidden
        layer nodes.  Then calculates the activations of the nodes in the output layer in the same manner.  Sets the
        .hidden_act and .output_act attributes of the NeuralNetwork object to equal the newly calculated activations
        of the nodes in the hidden and output layers respectively, given the current input activation.

            Params:
                input_act - an array whose values are the activations of the input layer nodes. This represents
                            an input example, whose output is to be predicted

            Returns:
                None (but this method does set the .hidden_act and .output_act attributes)
        """

        self.input_act = input_act

        hidden_z = np.dot(input_act, self.hidden_weights) + self.hidden_biases

        hidden_act = activation(hidden_z, self.activation)

        output_z = np.dot(hidden_act, self.output_weights) + self.output_biases

        output_act = activation(output_z, self.activation)

        self.hidden_act = hidden_act
        self.output_act = output_act



    def backprop(self, target):
        """
        Given the target output, this method calculates the difference between the target output and the output
        acquired from passing the target's associated input into a forward pass through the network. Then, it calculates
        the derivative of the error with respect to each of the activations in the output and hidden layers. These derivatives
        are used to adjust the weights and biases connected to each of the nodes in the output and hidden layers in the direction
        that decreases the error.  The weights and biases of the output and hidden layers (attributes of the NeuralNetwork object)
        are adjusted by this method.

            Params:
                target - an array whose values are the desired activations of an associated input

            Returns:
                None (but this method does modify the .hidden_weights, .hidden_biases, .output_weights, and .output_biases
                        attributes' values)
        """
        
        error = target - self.output_act # div by batchsize if doing BGD

        o_gradient = activation_derivative(self.output_act, self.activation)

        h_gradient = activation_derivative(self.hidden_act, self.activation)

        delta_out = error * o_gradient

        delta_hidden = np.dot(delta_out, self.output_weights.T) * h_gradient

        self.output_weights += np.dot(self.hidden_act.T, delta_out) * self.lr

        self.hidden_weights += np.dot(self.input_act.T, delta_hidden) * self.lr

        self.output_biases += np.sum(delta_out, axis=0, keepdims=True) * self.lr

        self.hidden_biases += np.sum(delta_hidden, axis=0, keepdims=True) * self.lr


    def fit(self, verbose=False):
        """
        For each of self.iters iterations, this function shuffles the input data and its target
        outputs (self.x and self.y), then feeds the first self.batchSize of these examples into the .feedforward()
        and .backprop() methods. The weights and biases of the network are updated with each sample (stochastic
        gradient descent). The loss is calculated for each training example, and stored along with the iteration
        number and sample number in an array.  The average loss over the entire iteration is also stored in an array.

            Params:
                verbose - False by default. When true, will print the loss of the current sample, the iteration number,
                            and the sample number for each sample

            Returns:
                an array of tuples containing the loss, iteration number, and sample number for each sample that was
                    input into the .feedforward() method.
        """

        loss = 999
        loss_iter_sample = []
        epoch_losses = []

        for i in range(self.iters):
            i+=1

            losses = []

            shuffled = sklearn.utils.shuffle(self.x, self.y) # does not shuffle in place
            shuffled_x, shuffled_y = shuffled[0], shuffled[1]

            for s in range(self.batchSize): # range(len(shuffled_x)): to go through all examples in each iteration

                sample = shuffled_x[s]
                target = shuffled_y[s]

                sample = np.reshape(sample, (1,self.setup[0][0]))
                target = np.reshape(target, (1,self.setup[1][1]))

                self.feedforward(sample)
                self.backprop(target)

                prev_loss = loss
                loss = np.sum((target - self.output_act)**2)

                losses.append(loss)

                loss_iter_sample.append((loss, i, s))

                if verbose:
                    print('loss = %.2f. iter = %d. sample = %d' % (loss, i, s))

            epoch_losses.append(sum(losses) / len(losses))

        # Can return epoch_losses here to see loss per epoch
        return loss_iter_sample



        
    def predict(self, input_act):
        """
        Passes the given input activation array into the .feedforward() method.  If the NeuralNetwork object is
        an autoencoder, then this method returns an array of zeros and a single 1 at the index with the maximum
        output activation in the network's output layer.  If the NeuralNetwork is not an autoencoder, then this
        function simply returns the network's output activation array.

            Params:
                input_act - an array whose values are the activations of the input layer nodes. This represents
                            an input example, whose output is to be predicted

            Returns:
                Either an array of zeros with a 1 at the index of the output layer's maximum activation (if the
                network is an autoencoder), or the network's output activation array itself.
        """
        
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
    Takes an array and returns an array of the same shape after applying the specified activation function.

        Params:
            x - a numpy array
            activ - a string. Currently can only be 'sigmoid' or 'relu'

        Returns:
            an array with the same shape as x

    """
    if activ == 'sigmoid':
        return 1/(1+np.exp(-x))
    elif activ == 'relu':
        return np.maximum(0, x)
    else:
        raise Exception('Activation function "%s" is not supported' % activ)



def activation_derivative(x, activ):
    """
    Calculates and returns the derivative of the given activation function at the values in the given array.

        Params:
            x - a numpy array
            activ - a string. Currently can only be 'sigmoid' or 'relu'

        Returns:
            an array with the same shape as x

    """
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
    """
    Takes a string of dna sequence and returns the one-hot encoded form of the sequence.

        Params:
            seq - a string of As, Ts, Cs, and Gs

        Returns:
            a numpy array containing the one-hot encoded representation of the input sequence

    """
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
        else:
            raise Exception('Character in DNA sequence not supported')

    flat_arr = np.array(one_hot).flatten() # convenient way of combining a list of lists
    return np.array(flat_arr)


def make_training_examples(pos_file, neg_file):
    """
    Reads a .txt file with positive DNA sequences and a .fa file with negative sequences. Fills a
    dictionary with the extracted data.  The keys of the dictionary are the sequences, and the values are
    1s for positive sequences and 0s for negative sequences.  The dictionary contains 411 negative examples,
    and as many positive examples as are in the .txt file.  The negative examples are the first 17 characters from
    the start of lines containing sequences in the .fa file.  Lastly, the function ensures that there are no duplicate
    sequences, and if a sequence is both a positive and negative example, then it will be stored as a positive
    example only.

        Params:
            pos_file - a .txt file with positive sequences separated by line breaks
            neg_file - a .fa file with negative sequences separated by '>' characters

        Returns:
            a dictionary with keys = sequences, and values = 1s and 0s (1s for positive examples, 0s for negative)

    """

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
    """
    Makes 8 unique bitvectors, each of length 8, and each full of zeros besides a single 1.

        Params:
            None

        Returns:
            A list of 8 different, 8-element numpy arrays

    """
    unique_bvs = []

    for i in range(8):
        ex = []
        for j in range(8):
            if i == j:
                ex.append(1)
            else:
                ex.append(0)
        unique_bvs.append(np.array(ex))

    return unique_bvs







