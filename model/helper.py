import os
import sys
import time
import math
import random
import pickle
import numpy as np
import pandas as pd

# Helper method for loading the data split
def l(suffix='A'):
    p = '../data_split.pkl'
    with open(p, 'rb') as f:
        d = pickle.load(f)

    def m():
        t = np.append(d['X_%s'%suffix], d['Y_%s'%suffix][:,None], axis=1)
        return t

    tt = pd.DataFrame(m(), columns=('Artist', 'Track', 'User', 'Time', 'Rating'))
    return tt

# Splitting the data input and rating
def split(d):
    m = d.shape[0]
    X = d[:, 1:]
    y = d[:, [0]]
    return X, y

# Just a wrapper method to make settings easier
def wrapper(suffix='A', filters=None):
    if filters:
        df, _= load_dataset(val_ratio=0., shuffle=False, suffix='onehot', debug=True, suffix2=suffix)
        df = df.loc[:, ['Rating'] + filters]
        df = df.as_matrix().astype(float)
        X, y = split(df)
    else:
        X, y, _, _, _ = load_dataset(val_ratio=0., shuffle=False, suffix='onehot', suffix2=suffix)

    return X, y

# The real method that merges and preprocesses the data
def load_dataset(val_ratio=0., shuffle=False, suffix='onehot', debug=False, suffix2='A'):
    users_table_path = os.path.join(os.getcwd(), '../data/users_cleaned_%s.csv' % suffix)
    words_table_path = os.path.join(os.getcwd(), '../data/words_cleaned_%s.csv' % suffix)

    users_table = pd.read_csv(users_table_path)
    words_table = pd.read_csv(words_table_path)
    train_table = l(suffix2)

    all_table_innerjoin = train_table.merge(users_table, left_on='User', right_on='RESPID', how='inner')
    all_table_innerjoin = all_table_innerjoin.merge(words_table, left_on=['Artist', 'User'], right_on=['Artist', 'User'], how='inner')

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

    for col in range(2, 83):
        replacement = all_table_innerjoin.ix[all_table_innerjoin.ix[:,col]!=2.,col].mean()
        all_table_innerjoin.ix[all_table_innerjoin.ix[:,col]==2.,col] = replacement

    if debug:
        return all_table_innerjoin, None

    dataset = all_table_innerjoin.as_matrix().astype(float)
    dataset_size = dataset.shape[0]

    if shuffle:
        shuffledidx = np.arange(dataset_size)
        np.random.shuffle(shuffledidx)
        dataset = dataset[shuffledidx, :]

    train_ratio = 1. - val_ratio
    assert train_ratio > 0., "wtf?"
    train_size = int(math.ceil(train_ratio * dataset_size))

    X_train = dataset[:train_size, 1:]
    y_train = dataset[:train_size, [0]]

    X_val = dataset[train_size:, 1:]
    y_val = dataset[train_size:, [0]]

    return X_train, y_train, X_val, y_val, None