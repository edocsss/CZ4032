import os
import sys
import time
import math
import random
import pandas
import numpy as np
from getopt import getopt

import theano
import theano.tensor as T
import lasagne
from lasagne.init import *
from lasagne.layers import *
from lasagne.updates import *
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.regularization import *

def load_dataset(val_ratio=.20, shuffle=True, suffix='onehot'):
    users_table_path = os.path.join(os.getcwd(), '../data/users_cleaned_%s.csv' % suffix)
    words_table_path = os.path.join(os.getcwd(), '../data/words_cleaned_%s.csv' % suffix)
    train_table_path = os.path.join(os.getcwd(), '../data/train.csv')
    test_table_path = os.path.join(os.getcwd(), '../data/test.csv')
    users_table = pandas.read_csv(users_table_path)
    words_table = pandas.read_csv(words_table_path)
    train_table = pandas.read_csv(train_table_path)
    test_table = pandas.read_csv(test_table_path)
    users_words_innerjoin = pandas.merge(words_table, users_table, how='inner', left_on='User', right_on='RESPID')
    all_table_innerjoin = pandas.merge(train_table, users_words_innerjoin, how='inner', on=('User', 'Artist'))
    test_table_innerjoin = pandas.merge(test_table, users_words_innerjoin, how='left', on=('User', 'Artist'))
    # sapu bersih
    all_table_innerjoin.drop(
        axis=1,
        inplace=True,
        labels=[
            'Artist',
            'Track',
            'User',
            'Time',
            'Unnamed: 0_x',
            'Unnamed: 0_y',
            'HEARD_OF',
            'OWN_ARTIST_MUSIC',
            'RESPID',
            'GENDER',
            'AGE',
            'WORKING',
            'REGION',
            'MUSIC',
            ],
        )
    test_table_innerjoin.drop(
        axis=1,
        inplace=True,
        labels=[
            # 'Artist',
            # 'Track',
            # 'User',
            # 'Time',
            'Unnamed: 0_x',
            'Unnamed: 0_y',
            'HEARD_OF',
            'OWN_ARTIST_MUSIC',
            'RESPID',
            'GENDER',
            'AGE',
            'WORKING',
            'REGION',
            'MUSIC',
            ],
        )
    # sweeping
    for col in range(2, 83):
        replacement = all_table_innerjoin.ix[all_table_innerjoin.ix[:,col]!=2.,col].mean()
        all_table_innerjoin.ix[all_table_innerjoin.ix[:,col]==2.,col] = replacement
    # sweeping testset
    for col in range(4, test_table_innerjoin.shape[1]):
        sel = test_table_innerjoin.ix[:,col].isnull()
        if 5 <= col < 86:
            test_table_innerjoin.ix[sel,col] = 2.
            replacement = test_table_innerjoin.ix[test_table_innerjoin.ix[:,col]!=2.,col].mean()
            test_table_innerjoin.ix[test_table_innerjoin.ix[:,col]==2.,col] = replacement
        else:
            test_table_innerjoin.ix[sel,col] = test_table_innerjoin.ix[~sel,col].mean()
    test_table_innerjoin.drop(
        axis=1,
        inplace=True,
        labels=[
            'Artist',
            'Track',
            'User',
            'Time',
            ],
        )
    for col in range(1, 82):
        replacement = test_table_innerjoin.ix[test_table_innerjoin.ix[:,col]!=2.,col].mean()
        test_table_innerjoin.ix[test_table_innerjoin.ix[:,col]==2.,col] = replacement
    dataset = all_table_innerjoin.as_matrix().astype(float)
    dataset_size = dataset.shape[0]
    if shuffle:
        shuffledidx = np.arange(dataset_size)
        np.random.shuffle(shuffledidx)
        dataset = dataset[shuffledidx, :]
    train_ratio = 1. - val_ratio
    assert train_ratio > 0., "wtf?"
    train_size = int(math.ceil(train_ratio * dataset_size))
    val_size = dataset_size - train_size
    X_train = dataset[:train_size, 1:]
    y_train = dataset[:train_size, [0]]
    X_val = dataset[train_size:, 1:]
    y_val = dataset[train_size:, [0]]
    X_test = test_table_innerjoin.as_matrix().astype(float)
    return X_train, y_train, X_val, y_val, X_test

if __name__ == '__main__':
    # random.seed(42)
    # np.random.seed(42)

    hidden_size = 20
    Alpha = 3e-3 # Learning rate
    Beta = 1e-6 # L2 Reg
    max_iter = 1000
    suffix = 'onehot'
    print('Import done! Parsing parameters...')
    args = sys.argv[1:]
    opts, args = getopt(args, '', ['max-iter=', 'Alpha=', 'Beta=', 'suffix=', 'hidden-size='])
    for option, value in opts:
        if option == '--max-iter':
            max_iter = int(value)
        elif option == '--Alpha':
            Alpha = float(value)
        elif option == '--Beta':
            Beta = float(value)
        elif option == '--suffix':
            suffix = value
        elif option == '--hidden-size':
            hidden_size = int(value)

    print('Parsing done! Loading dataset...')
    X_train, y_train, X_val, y_val, X_test = load_dataset(suffix=suffix)
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    # y_train /= 100.
    # y_val /= 100.

    print('Dataset loaded! Creating ANN...')
    input_var = T.matrix('inputs')
    target_var = T.col('targets')
    input_size = X_train.shape[1]
    output_size = 1
    input_layer = InputLayer(
        shape=(None, input_size),
        input_var=input_var,
        name='input layer'
        )
    hidden_layer = DenseLayer(
        input_layer,
        num_units=hidden_size,
        nonlinearity=linear,
        W=Uniform(range=(-0.05, 0.05)),
        b=Uniform(range=(-0.05, 0.05)),
        name='hidden layer'
        )
    output_layer = DenseLayer(
        hidden_layer,
        num_units=output_size,
        nonlinearity=linear,
        W=Uniform(range=(-0.05, 0.05)),
        b=Uniform(range=(-0.05, 0.05)),
        name='output layer'
        )
    neural_network = output_layer

    # For training
    parameters = get_all_params(neural_network, trainable=True)
    prediction = get_output(neural_network)
    mse = squared_error(prediction, target_var).mean()
    # mcc = binary_crossentropy(prediction, target_var).mean()
    reg = Beta * regularize_layer_params(neural_network, l2)
    loss = mse + reg
    # loss = mcc + reg
    updates = sgd(loss, parameters, learning_rate=Alpha)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # For CV and testing
    test_fn = theano.function([input_var, target_var], loss)
    predict_fn = theano.function([input_var], prediction)
    print('ANN created! Initiate training...')

    initial_params = get_all_param_values(neural_network)
    best_val_err = 1e99 # Capture best cv err
    best_params = None # and best params
    print('Training start...')
    training_start_time = time.time()
    for it in range(max_iter):
        train_err = train_fn(X_train, y_train)
        val_err = test_fn(X_val, y_val)
        if val_err < best_val_err:
            best_val_err = val_err
            best_params = get_all_param_values(neural_network)
    print('Training duration: %.4f' % (time.time() - training_start_time))
    print('Training loss: %.9f, val loss: %.9f, best val err: %.9f' % (train_err, val_err, best_val_err))

    result = predict_fn(X_test)
