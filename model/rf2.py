import os
import sys
import time
import math
import random
import pickle
import numpy as np
import pandas as pd
from helper import load_dataset
from sklearn.ensemble import RandomForestRegressor as RFG

def split(d, val_ratio=.3, shuffle=True):
    m = d.shape[0]
    if shuffle:
        shuffled_idx = np.arange(m)
        np.random.shuffle(shuffled_idx)
        d = d[shuffled_idx]
    train_ratio = 1. - val_ratio
    assert train_ratio > 0., 'wtf?'
    train_size = int(math.ceil(train_ratio * m))
    val_size = m - train_size
    X_train = d[:train_size, 1:]
    y_train = d[:train_size, [0]]
    X_val = d[train_size:, 1:]
    y_val = d[train_size:, [0]]
    return X_train, y_train, X_val, y_val

if __name__ == '__main__':
    # RF by User demographics (non-Q) [A, Tr, U, Demo] [Peter]
    d, t = load_dataset(debug=True)
    col_filters = ['GENDER_0', 'GENDER_1', 'WORKING_0',
        'WORKING_1', 'WORKING_2', 'WORKING_3', 'WORKING_4', 'WORKING_5',
        'WORKING_6', 'WORKING_7', 'WORKING_8', 'WORKING_9', 'WORKING_10',
        'WORKING_11', 'WORKING_12', 'WORKING_13', 'REGION_0', 'REGION_1',
        'REGION_2', 'REGION_3', 'REGION_4', 'REGION_5', 'REGION_6', 'MUSIC_0',
        'MUSIC_1', 'MUSIC_2', 'MUSIC_3', 'MUSIC_4', 'MUSIC_5', 'AGE_RANGE_0-15',
        'AGE_RANGE_16-25', 'AGE_RANGE_26-35', 'AGE_RANGE_36-45',
        'AGE_RANGE_46-55', 'AGE_RANGE_56-65', 'AGE_RANGE_66-']
    train_df = d.loc[:, ['Rating'] + col_filters]
    test_df = t.loc[:, col_filters]
    dataset = train_df.as_matrix().astype(float)
    X_train, y_train, X_val, y_val = split(dataset, val_ratio=0., shuffle=False)
    # For fitting into RF
    y_train = y_train.flatten()
    y_val = y_val.flatten()
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    X_test = test_df.as_matrix().astype(float)

    print('Saving RAM')
    # For the love of my laptop, please save the RAM
    del train_df
    del test_df
    print('RAM saved!')

    print('Dataset loaded! Training with RFG...')
    rf = RFG(n_estimators=38)
    rf.fit(X_train, y_train)
    with open('rf2.pkl', 'wb') as f:
        pickle.dump(rf,f)
    print('Training done!')
    y_hat=rf.predict(X_train)
    with open('rf2_predicts.pkl', 'wb') as f:
        pickle.dump(np.append(y_hat[:,None], y_train[:,None], axis=1), f)
