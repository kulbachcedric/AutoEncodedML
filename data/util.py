import numpy as np


def get_train_test_indices(data, test_size=0.2):
    n_train_samples = np.round((1 - test_size) * data.shape[0]).astype(int)
    permutation = np.random.permutation(data.shape[0])
    indices_train = permutation[:n_train_samples]
    indices_test = permutation[n_train_samples:]
    return indices_train, indices_test


def corrupt_with_mask(data, noise_level=0.5, pepper=True):
    num_zeros = round(noise_level * np.prod(data.shape[1:]))
    out = np.copy(data)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    for sample in out:
        indices = np.array([np.random.choice(np.arange(i), num_zeros, replace=False) for i in sample.shape])
        sample[indices] = np.where(np.random.randint(0, 2) == 1, mins[indices], maxs[indices]) if pepper else mins[
            indices]
    return out


def corrupt_gaussian(data, noise_level=0.5):
    std = np.diag(np.std(data, axis=0))
    mean = np.zeros(data.shape[1:])
    noise = np.random.multivariate_normal(mean=mean, size=data.shape[0], cov=std)
    return data + noise_level * noise

