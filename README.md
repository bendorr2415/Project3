# Project 3 - Neural Networks
## Due 03/19/2021

![BuildStatus](https://github.com/bendorr2415/Project3/workflows/HW3/badge.svg?event=push)

### main
Runs all code in scripts/\_\_main\_\_.py, useful for part 2
```
python -m scripts
```

### testing
Testing is as simple as running
```
python -m pytest test/*
```
from the root directory of this project.

### Layout of Repo

```
-Ben_Orr_BMI203_HW3: a Jupyter notebook containing written answers to assignment questions in prose, as well as function definitions and function
calls that were used in answering assignment questions.

-scripts/
  - NN.py: contains the NeuralNetwork class definition, as well as activation, activation_derivative, one_hot_encode_dna, make_training_examples, and make_autoencoder_training_examples function definitions.

-test/
  - test_NN.py: contains function definitions for unit tests

-data/
  - rap1-lieb-positives.txt: 17-base-long dna sequences representing Rap1 binding sites
  - rap1-lieb-test.txt: 17-base-long dna sequences that have no label indicating whether they're Rap1 binding sites
  - yeast-upstream-1k-negative.fa: 1000-base-long dna sequences representing regions of the yeast genome that are 1k bases upstream of yeast genes. The vast majority of these sequences should not be Rap1 binding sites.

```

### API

```
Class NeuralNetwork():

  - Attributes:

    - x: a list of arrays, representing input layer activations
    - y: a list of arrays, representing the target output layer activations of each of the x's (inputs)
    - setup: a list of lists. Each sublist contains two values, representing the size of the previous layer and the size of the next layer.  Layer sizes must align between sublists. For example, for an 8 x 3 x 8 network, setup=[[8,3][3,8]].
    - activation: a string that indicates the activation function to be used for each node in the network. Currently supports 'sigmoid' or 'relu'
    - batchSize: an int indicating the number of samples with which to perform stochastic gradient descent for each iteration in the training loop in the .fit() method.
    - is_autoencoder: a boolean. True if the network is an autoencoder.
    - lr (learning rate): a float that determines the magnitude of the adjustments made to the weights and biases of the network for each step taken by the network.
    - seed: an int used to set a random seed for reproducibility of results.
    - iters: the number of iterations for the training loop in the .fit() method to undergo.


  - Methods:
  

    - feedforward (self, input_act):

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
                
                
                
    - backprop (self, target):

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
                        
                        
                        
    - fit (self, verbose=False):
    
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
                    
                    
                    
    - predict (self, input_act):

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

```
