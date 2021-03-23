
import pytest

from scripts import NN


def test_encoder():
    assert True

def test_encoder_relu():
    assert True

def test_one_d_ouput():
    assert True

seq = 'actgactg'
def one_hot_encode_test(seq):
    one_hot = NN.one_hot_encode_dna(seq)
    print(one_hot)