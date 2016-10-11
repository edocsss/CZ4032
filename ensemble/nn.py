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

def lr_by_artist():
    filename='../edwin/lr_by_artist/lr_by_artist_training_predictions_result.zip'
    y=np.array(pickle.load(gzip.GzipFile(filename)))
    return y
def rf_by_artist():
    filename='../edwin/rf_by_artist/rf_by_artist_training_predictions_result.zip'
    y=np.array(pickle.load(gzip.GzipFile(filename)))
    return y
def rf_full():
    filename='../edwin/rf_full/rf_full_training_predictions_result.zip'
    y=np.array(pickle.load(gzip.GzipFile(filename)))
    return y
def gbr():
    filename='../kenrick/gbr_preds.pkl'
    with open(filename, 'rb') as f:
        y=np.array(pickle.load(f))
    return y
def nn():
    filename='../model/nn_predicts.pkl'
    with open(filename, 'rb') as f:
        y=pickle.load(f)
    y=y[:,] # swap
    return y
def rf1():
    filename='../model/rf1_predicts.pkl'
    with open(filename, 'rb') as f:
        y=pickle.load(f)
    return y
def rf2():
    filename='../model/rf2_predicts.pkl'
    with open(filename, 'rb') as f:
        y=pickle.load(f)
    return y
def lasso():
    filename='../martinus/lasso_prediction.pkl'
    with open(filename, 'rb') as f:
        y=np.array(pickle.load(f))
    return y
def linear_reg():
    filename='../martinus/linear_regression_prediction.pkl'
    y=np.zeros((0,2))
    with open(filename, 'rb') as f:
        p=pickle.load(f)
    for pre,exp in p:
        y=np.append(y,
            np.append(pre[:,None],exp[:,None],axis=1),
            axis=0)
    return y
def ridge():
    filename='../martinus/ridge_prediction.pkl'
    y=np.zeros((0,2))
    with open(filename, 'rb') as f:
        p=pickle.load(f)
    for pre,exp in p:
        y=np.append(y,
            np.append(pre[:,None],exp[:,None],axis=1),
            axis=0)
    return y
print('import selesai')
handlers=[
    lr_by_artist,
    rf_by_artist,
    rf_full,
    gbr,
    nn,
    rf1,
    rf2,
    lasso,
    linear_reg,
    ridge,
    ]

ys = []
idcs=np.zeros(len(handlers),dtype=int)
max_idcs=np.zeros(0,dtype=int)
for handler in handlers:
    y=handler()
    y=y[y[:,1].argsort()]
    ys.append(y)
    max_idcs=np.append(max_idcs,len(y))

X_train = np.zeros((0,len(handlers)))
y_train = np.zeros(0)
while (idcs < max_idcs).all():
    vals=np.zeros(len(idcs))
    vals2=np.zeros(len(idcs))
    for i in range(len(idcs)):
        vals[i]=ys[i][idcs[i],1]
        vals2[i]=ys[i][idcs[i],0]
    if vals.max() == vals.min():
        X_train=np.append(X_train,vals2[None,:],axis=0)
        y_train=np.append(y_train,[vals[0]],axis=0)
        idcs+=1
    else:
        idcs[vals.argmin()] += 1
print('loaded')
train_size=X_train.shape[0]
shuffled_idx = np.arange(train_size)
np.random.shuffle(shuffled_idx)
X_train = X_train[shuffled_idx]

input_size = X_train.shape[1]
output_size = 1
val_size=int(0.3*train_size)
X_val=X_train[-val_size:]
y_val=y_train[-val_size:][:,None]
X_train=np.resize(X_train,(train_size-val_size,len(handlers)))
y_train=np.resize(y_train,(train_size-val_size))[:,None]
train_size-=val_size
X_mean=X_train.mean(axis=0)
X_std=X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std

hidden_size = 20
Alpha = 1e-3 # Learning rate
Beta = 0. # L2 Reg
max_iter = 1000
max_tol = 10

input_var = T.matrix('inputs')
target_var = T.col('targets')

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
# mcc = binary_crossentropy(prediction, target_var).mean()
reg = Beta * regularize_layer_params(neural_network, l2)
loss = mse + reg
# loss = mcc + reg
updates = sgd(loss, parameters, learning_rate=Alpha)
train_fn = theano.function([input_var, target_var], loss, updates=updates)
# For CV and testing
test_fn = theano.function([input_var, target_var], loss)
predict_fn = theano.function([input_var], prediction)

n_tol = 0
best_val_err = test_fn(X_val, y_val) # Capture best cv err
best_params = get_all_param_values(neural_network) # and best params
print('Training start...')
training_start_time = time.time()
for it in range(max_iter):
    train_err = train_fn(X_train, y_train)
    val_err = test_fn(X_val, y_val)
    # print(train_err, val_err)
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

result=predict_fn(X_val)

with open('nn_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)
print('params dumped')
