import gzip
import numpy as np
import pprint
from sklearn.metrics import mean_squared_error
import math

from model.helper import *

# Load the results of linear regression by artist models
def lr_by_artist(suffix='C'):
    filename='../edwin/lr_by_artist/lr_by_artist_training_predictions_result_{}.zip'.format(suffix)
    y=np.array(pickle.load(gzip.GzipFile(filename)))
    return y


# Load the results of Random Forest by artist models
def rf_by_artist(suffix='C'):
    filename='../edwin/rf_by_artist/rf_by_artist_training_predictions_result_{}.zip'.format(suffix)
    y=np.array(pickle.load(gzip.GzipFile(filename)))
    return y


# Load the results of Random Forest using the full data features
def rf_full(suffix='C'):
    filename='../edwin/rf_full/rf_full_training_predictions_result_{}.zip'.format(suffix)
    y=np.array(pickle.load(gzip.GzipFile(filename)))
    return y


# Load the results of Gradient Boosting Regression using the full data features
def gbr(suffix='C'):
    filename='../kenrick/gbr_preds_{}.pkl'.format(suffix)
    with open(filename, 'rb') as f:
        y=np.array(pickle.load(f))
    return y


# Load the results of the individual Neural Network using the full data features
def nn(suffix='C'):
    filename='../model/nn_y_{}_hat.pkl'.format(suffix)
    with open(filename, 'rb') as f:
        y=pickle.load(f)
    y=y[:,] # swap
    return y


# Load the results of the Random Forest by User Demographics data
def rf1(suffix='C'):
    filename='../model/rf1_y_{}_hat.pkl'.format(suffix)
    with open(filename, 'rb') as f:
        y=pickle.load(f)
    return y


# Load the results of the Random Forest by Questions and Words data
def rf2(suffix='C'):
    filename='../model/rf2_y_{}_hat.pkl'.format(suffix)
    with open(filename, 'rb') as f:
        y=pickle.load(f)
    return y


# Load the results of the Lasso Regression model using the full data features
def lasso(suffix='C'):
    filename='../martinus/lasso_prediction_{}.pkl'.format(suffix)
    with open(filename, 'rb') as f:
        y=np.array(pickle.load(f))
    return y


# Load the results of the Linear Regression model using the full data features
def linear_reg(suffix='C'):
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
def ridge(suffix='C'):
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
# Remove methods if needed in order to get the best 3 models (GBR full, RF full, RF by artist ID)
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

# Reading predicted ratings from each individual model
full_data = []
for handler in handlers:
    data = handler(suffix='C')
    full_data.append(data.tolist())

X_C = []
Y_C = np.array([item[1] for item in full_data[0]])

for i in range(len(full_data[0])):
    x = [item[i][0] for item in full_data]
    X_C.append(x)

X_C = np.array(X_C)
print('Data loaded!')


# Do a simple averaging on the predicted ratings from each individual model
Y_pred = [np.mean(x) for i, x in enumerate(X_C)]
print(mean_squared_error(Y_pred, Y_C), math.sqrt(mean_squared_error(Y_pred, Y_C)))