
import pytest
import numpy as np

from scripts import NN


def test_one_hot_encode():
	seq = 'actgactg'
	one_hot = NN.one_hot_encode_dna(seq)
	assert one_hot.all() == np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]).all()

def test_weight_matrix_dimensions():
	training_examples = NN.make_autoencoder_training_examples()
	training_examples = np.array(training_examples)
	x = training_examples
	y = training_examples

	setup = [[8,3],[3,8]]
	batchSize = 8
	activation = 'sigmoid'

	nn = NN.NeuralNetwork(x,y,setup,activation,batchSize,is_autoencoder=True,lr=.01,seed=1,iters=15000)

	assert nn.hidden_weights.shape == (8,3)
	assert nn.output_weights.shape == (3,8)

def test_weights_update():
	training_examples = NN.make_autoencoder_training_examples()
	training_examples = np.array(training_examples)
	x = training_examples
	y = training_examples

	setup = [[8,3],[3,8]]
	batchSize = 8
	activation = 'sigmoid'

	nn = NN.NeuralNetwork(x,y,setup,activation,batchSize,is_autoencoder=True,lr=.01,seed=1,iters=100)
	nn.fit()
	w1 = np.array(nn.hidden_weights, copy=True)
	nn.fit()
	w2 = np.array(nn.hidden_weights, copy=True)

	for i in range(len(w1[0])):
		assert not w1[0][i] == w2[0][i]

def test_autoencoder():
	training_examples = NN.make_autoencoder_training_examples()
	training_examples = np.array(training_examples)
	x = training_examples
	y = training_examples

	setup = [[8,3,"sigmoid"],[3,8,"sigmoid"]]
	batchSize = 8
	activation = 'sigmoid'

	nn = NN.NeuralNetwork(x,y,setup,activation,batchSize,is_autoencoder=True,lr=.01,seed=1,iters=15000)
	loss_iter_sample = nn.fit()

	data = []
	for i in range(8):
		target = [0,0,0,0,0,0,0,0]
		target[i] = 1
		out = nn.predict(target)
		assert np.array(target).all() == out.all()
