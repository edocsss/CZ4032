import os
import sys
import time
import math
import pickle
import random
import numpy as np
import pandas as pd
from getopt import getopt
from model.helper import *

import theano
import theano.tensor as T
import lasagne
from lasagne.init import *
from lasagne.layers import *
from lasagne.updates import *
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.regularization import *

if __name__ == '__main__':
    # Hyper-parameters initialization
    hidden_size = 50
    Alpha = 1e-3 # Learning rate
    Beta = 0. # L2 Reg
    max_iter = 1000
    max_tol = 10
    suffix = 'onehot'
    print('Import done! Parsing parameters...')

    # Loading dataset
    print('Parsing done! Loading dataset...')
    X_train, y_train, X_val, y_val, X_test = load_dataset(val_ratio=0.3,suffix=suffix,shuffle=False)
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    # X_test = (X_test - X_mean) / X_std

    print('Dataset loaded! Creating ANN...')
    input_size = X_train.shape[1]
    output_size = 1
    input_var = T.matrix('inputs')
    target_var = T.col('targets')

    # Instantiating the neural network
    input_layer = InputLayer(
        shape=(None, input_size),
        input_var=input_var,
        name='input layer'
        )
    hidden_layer = DenseLayer(
        input_layer,
        num_units=hidden_size,
        nonlinearity=linear,
        W=Uniform(-0.05),
        b=Uniform(-0.05),
        name='hidden layer'
        )
    output_layer = DenseLayer(
        hidden_layer,
        num_units=output_size,
        nonlinearity=linear,
        W=Uniform(-0.05),
        b=Uniform(-0.05),
        name='output layer'
        )
    neural_network = output_layer

    # Preparing training and testing functions
    # For training
    parameters = get_all_params(neural_network, trainable=True)
    prediction = get_output(neural_network)
    mse = squared_error(prediction, target_var).mean()
    reg = Beta * regularize_layer_params(neural_network, l2)
    loss = mse + reg
    updates = sgd(loss, parameters, learning_rate=Alpha)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # For CV and testing
    test_fn = theano.function([input_var, target_var], loss)
    predict_fn = theano.function([input_var], prediction)
    print('ANN created! Initiate training...')

    # Getting ready to train the network
    n_tol = 0
    best_val_err = test_fn(X_val, y_val) # Capture best cv err
    best_params = get_all_param_values(neural_network) # and best params
    print('Training start...')
    training_start_time = time.time()

    # Training
    for it in range(max_iter):
        train_err = train_fn(X_train, y_train)
        val_err = test_fn(X_val, y_val)

        if val_err < best_val_err:
            best_val_err = val_err
            best_params = get_all_param_values(neural_network)
            n_tol = 0
        else:
            n_tol = n_tol + 1

        if n_tol == max_tol:
            print('>>> Validation err is no longer improving')
            break

    print('Training duration: %.4f (%d epoch out of %d)' % (time.time() - training_start_time, it + 1, max_iter))
    print('Training loss: %.9f, val loss: %.9f, best val err: %.9f' % (train_err, val_err, best_val_err))

    # Saving model to pickle file
    with open('nn_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    # Predicting
    X_B, y_B = wrapper(suffix='B')
    y_B_hat = predict_fn(X_B)
    with open('nn_y_B_hat.pkl', 'wb') as f:
        pickle.dump(np.append(y_B_hat[:,None], y_B[:,None], axis=1), f)