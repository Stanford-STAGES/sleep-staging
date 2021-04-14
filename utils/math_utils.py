import numpy as np


def softmax(x, axis=-1):
    s = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=axis, keepdims=True)
    return e_x / div