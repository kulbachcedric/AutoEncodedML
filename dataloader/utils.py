import numpy as np

def get_train_test_indices(data, test_size=0.2):
  n_train_samples = np.round((1 - test_size) * data.shape[0]).astype(int)
  permutation = np.random.permutation(data.shape[0])
  indices_train = permutation[:n_train_samples]
  indices_test = permutation[n_train_samples:]
  return indices_train, indices_test
