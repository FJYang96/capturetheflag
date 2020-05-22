import numpy as np

def softmax(array, sharpening=1):
    '''
    Given an array, return a normalized softmax probability of choosing each item
    The formula is given by
            A[i] = exp(A[i] * sharpening) / sum_{j}{exp(A[j] * sharpening)}
    '''
    a_sharp = array * sharpening
    a_sharp_norm = a_sharp - a_sharp.max()
    a_exp = np.exp(a_sharp_norm)
    return a_exp / a_exp.sum()
