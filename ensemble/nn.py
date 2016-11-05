import os
import sys
import math
import time
import gzip
import random
import pickle
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import lasagne
from lasagne.init import *
from lasagne.layers import *
from lasagne.updates import *
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.regularization import *
import pprint
from model.helper import *
from sklearn.metrics import mean_squared_error

# Load the results of linear regression by artist models
def lr_by_artist(suffix='C_NN'):
    filename='../edwin/lr_by_artist/lr_by_artist_training_predictions_result_{}.zip'.format(suffix)
    y=np.array(pickle.load(gzip.GzipFile(filename)))
    return y


# Load the results of Random Forest by artist models
def rf_by_artist(suffix='C_NN'):
    filename='../edwin/rf_by_artist/rf_by_artist_training_predictions_result_{}.zip'.format(suffix)
    y=np.array(pickle.load(gzip.GzipFile(filename)))
    return y


# Load the results of Random Forest using the full data features
def rf_full(suffix='C_NN'):
    filename='../edwin/rf_full/rf_full_training_predictions_result_{}.zip'.format(suffix)
    y=np.array(pickle.load(gzip.GzipFile(filename)))
    return y


# Load the results of Gradient Boosting Regression using the full data features
def gbr(suffix='C_NN'):
    filename='../kenrick/gbr_preds_{}.pkl'.format(suffix)
    with open(filename, 'rb') as f:
        y=np.array(pickle.load(f))
    return y


# Load the results of the individual Neural Network using the full data features
def nn(suffix='C_NN'):
    filename='../model/nn_y_{}_hat.pkl'.format(suffix)
    with open(filename, 'rb') as f:
        y=pickle.load(f)
    y=y[:,] # swap
    return y


# Load the results of the Random Forest by User Demographics data
def rf1(suffix='C_NN'):
    filename='../model/rf1_y_{}_hat.pkl'.format(suffix)
    with open(filename, 'rb') as f:
        y=pickle.load(f)
    return y


# Load the results of the Random Forest by Questions and Words data
def rf2(suffix='C_NN'):
    filename='../model/rf2_y_{}_hat.pkl'.format(suffix)
    with open(filename, 'rb') as f:
        y=pickle.load(f)
    return y


# Load the results of the Lasso Regression model using the full data features
def lasso(suffix='C_NN'):
    filename='../martinus/lasso_prediction_{}.pkl'.format(suffix)
    with open(filename, 'rb') as f:
        y=np.array(pickle.load(f))
    return y


# Load the results of the Linear Regression model using the full data features
def linear_reg(suffix='C_NN'):
    filename='../martinus/linear_regression_prediction_{}.pkl'.format(suffix)
    y=np.zeros((0,2))
    with open(filename, 'rb') as f:
        p=pickle.load(f)
    for pre,exp in p:
        y=np.append(y,
            np.append(pre[:,None],exp[:,None],axis=1),
            axis=0)
    return y


# Load the results of the Ridge Regression model using the full data features
def ridge(suffix='C_NN'):
    filename='../martinus/ridge_prediction_{}.pkl'.format(suffix)
    y=np.zeros((0,2))
    with open(filename, 'rb') as f:
        p=pickle.load(f)
    for pre,exp in p:
        y=np.append(y,
            np.append(pre[:,None],exp[:,None],axis=1),
            axis=0)
    return y


# List of methods that will be called later
handlers=[
    # lr_by_artist,
    rf_by_artist,
    rf_full,
    gbr,
    # nn,
    # rf1,
    # rf2,
    # lasso,
    # linear_reg,
    # ridge,
]

# Reading all training data with the given suffix
full_data = []
for handler in handlers:
    data = handler(suffix='B')
    full_data.append(data.tolist())

X_train = []
y_train = np.array([item[1] for item in full_data[0]])

for i in range(len(full_data[0])):
    x = [item[i][0] for item in full_data]
    X_train.append(x)

X_train = np.array(X_train)
print('Data loaded!')


# Preparing training data
train_size = X_train.shape[0]
shuffled_idx = np.arange(train_size)
np.random.shuffle(shuffled_idx)
X_train = X_train[shuffled_idx]
y_train = y_train[shuffled_idx]

input_size = X_train.shape[1]
output_size = 1
val_size = int(0.3 * train_size)

X_val = X_train[-val_size:]
y_val = y_train[-val_size:][:, None]

X_train = np.resize(X_train,(train_size - val_size, len(handlers)))
y_train = np.resize(y_train,(train_size - val_size))[:, None]

train_size -= val_size
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std



# Neural Network Configuration
hidden_size = 30
Alpha = 1e-3 # Learning rate
Beta = 0.5 # L2 Reg
max_iter = 1000
max_tol = 10

input_var = T.matrix('inputs')
target_var = T.col('targets')

# Preparing NN layer
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
print('nn rdy')

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

# Start NN training
n_tol = 0
best_val_err = test_fn(X_val, y_val) # Capture best cv err
best_params = get_all_param_values(neural_network) # and best params
print('Training start...')
training_start_time = time.time()

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

result = predict_fn(X_val)
with open('nn_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)



# Read data using the handler functions
full_data = []
for handler in handlers:
    data = handler(suffix='C_NN')
    full_data.append(data.tolist())

X_C = []
Y_C = np.array([item[1] for item in full_data[0]])

for i in range(len(full_data[0])):
    x = [item[i][0] for item in full_data]
    X_C.append(x)

X_C = np.array(X_C)
X_C = (X_C - X_mean) / X_std

# Predict the final rating using the trained NN ensemble
Y_C_hat = predict_fn(X_C)

# Calculate MSE
mse = mean_squared_error(Y_C, Y_C_hat)
print(mse, math.sqrt(mse))