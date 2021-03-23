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
