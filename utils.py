#!/usr/bin/python
# -*- coding: utf-8 -*-

def zero_mean_normalization(np_matrix):
    """
    Given a numpy matrix, this function will normalize each columns to standard normal distribution
    i.e. for each columns, z = (x - mean) / std_deviation
    """
    means = np.mean(np_matrix, axis=0)
    std_deviations = np.std(np_matrix, axis=0)
    return (np_matrix - means) / std_deviations
