import numpy as np


def kl_divergence(x, y, axis=-1, epsilon=1e-10):
    return np.sum(x*np.log(x/(y+epsilon)+epsilon), axis=axis)


def category_2_one_hot(targets, nb_class):
    targets_one_hot = np.zeros(shape=(len(targets), nb_class))
    targets_one_hot[np.arange(len(targets)), targets] = 1.
    return targets_one_hot


def entropy(x):
    x = np.array(x)
    sum = np.sum(x)
    p = x*1.0/sum
    return -np.sum(p * np.log10(p))


def kl_var(x):
    mean = np.exp(np.mean(np.log(x+1e-10), axis=0, keepdims=True))
    mean = mean/np.sum(mean, axis=-1, keepdims=True)
    var = kl_divergence(mean, x)
    return np.mean(var)