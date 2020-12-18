import numpy as np


def get_train_test_indices(data, test_size=0.2):
    n_train_samples = np.round((1 - test_size) * data.shape[0]).astype(int)
    permutation = np.random.permutation(data.shape[0])
    indices_train = permutation[:n_train_samples]
    indices_test = permutation[n_train_samples:]
    return indices_train, indices_test


def corrupt_zero_mask(data, scale=0.5):
    data_masked = []
    for sample in data:
        sample_flat = sample.flatten()
        zeros = np.zeros_like(sample_flat)
        indices = np.random.choice(np.arange(zeros.size), replace=False, size=int(zeros.size * (1. - scale)))
        zeros[indices] = sample_flat[indices]
        sample_masked = np.reshape(zeros, sample.shape)
        data_masked.append(sample_masked)
    return np.array(data_masked)


def corrupt_gaussian(data, scale=0.5):
    return data + np.random.normal(size=data.shape, scale=scale)
