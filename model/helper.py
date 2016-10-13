import os
import sys
import time
import math
import random
import pickle
import numpy as np
import pandas as pd

# 132083
# 56607
def l(suffix='A'):
    p='../data_split.pkl'
    with open(p, 'rb') as f:
        d=pickle.load(f)
    def m():
        t=np.append(d['X_%s'%suffix],d['Y_%s'%suffix][:,None],axis=1)
        return t
    tt=pd.DataFrame(m(),
        columns=('Artist','Track','User','Time','Rating'),
        )
    return tt

def split(d):
    m = d.shape[0]
    X = d[:, 1:]
    y = d[:, [0]]
    return X, y

def wrapper(suffix='A', filters=None):
    if filters:
        df, _= load_dataset(val_ratio=0., shuffle=False, suffix='onehot', debug=True, suffix2=suffix)
        df = df.loc[:, ['Rating'] + filters]
        df = df.as_matrix().astype(float)
        X, y = split(df)
    else:
        X, y, _, _, _ = load_dataset(val_ratio=0., shuffle=False, suffix='onehot', suffix2=suffix)
    return X, y

def load_dataset(val_ratio=.30, shuffle=False, suffix='onehot', debug=False, suffix2='A'):
    users_table_path = os.path.join(os.getcwd(), '../data/users_cleaned_%s.csv' % suffix)
    words_table_path = os.path.join(os.getcwd(), '../data/words_cleaned_%s.csv' % suffix)
    # train_table_path = os.path.join(os.getcwd(), '../data/train.csv')
    # test_table_path = os.path.join(os.getcwd(), '../data/test.csv')
    users_table = pd.read_csv(users_table_path)
    words_table = pd.read_csv(words_table_path)
    train_table = l(suffix2)
    # test_table = pd.read_csv(test_table_path)
    users_words_innerjoin = pd.merge(words_table, users_table, how='inner', left_on='User', right_on='RESPID')
    all_table_innerjoin = pd.merge(train_table, users_words_innerjoin, how='inner', on=('User', 'Artist'))
    # test_table_innerjoin = pd.merge(test_table, users_words_innerjoin, how='left', on=('User', 'Artist'))
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
    # test_table_innerjoin.drop(
    #     axis=1,
    #     inplace=True,
    #     labels=[
    #         # 'Artist',
    #         # 'Track',
    #         # 'User',
    #         # 'Time',
    #         'Unnamed: 0_x',
    #         'Unnamed: 0_y',
    #         'HEARD_OF',
    #         'OWN_ARTIST_MUSIC',
    #         'RESPID',
    #         'GENDER',
    #         'AGE',
    #         'WORKING',
    #         'REGION',
    #         'MUSIC',
    #         ],
    #     )
    # sweeping
    for col in range(2, 83):
        replacement = all_table_innerjoin.ix[all_table_innerjoin.ix[:,col]!=2.,col].mean()
        all_table_innerjoin.ix[all_table_innerjoin.ix[:,col]==2.,col] = replacement
    # sweeping testset
    # for col in range(4, test_table_innerjoin.shape[1]):
    #     sel = test_table_innerjoin.ix[:,col].isnull()
    #     if 5 <= col < 86:
    #         test_table_innerjoin.ix[sel,col] = 2.
    #         replacement = test_table_innerjoin.ix[test_table_innerjoin.ix[:,col]!=2.,col].mean()
    #         test_table_innerjoin.ix[test_table_innerjoin.ix[:,col]==2.,col] = replacement
    #     else:
    #         test_table_innerjoin.ix[sel,col] = test_table_innerjoin.ix[~sel,col].mean()
    # test_table_innerjoin.drop(
    #     axis=1,
    #     inplace=True,
    #     labels=[
    #         'Artist',
    #         'Track',
    #         'User',
    #         'Time',
    #         ],
    #     )
    # for col in range(1, 82):
    #     replacement = test_table_innerjoin.ix[test_table_innerjoin.ix[:,col]!=2.,col].mean()
    #     test_table_innerjoin.ix[test_table_innerjoin.ix[:,col]==2.,col] = replacement
    if debug:
        # return all_table_innerjoin, test_table_innerjoin
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
    val_size = dataset_size - train_size
    X_train = dataset[:train_size, 1:]
    y_train = dataset[:train_size, [0]]
    X_val = dataset[train_size:, 1:]
    y_val = dataset[train_size:, [0]]
    # X_test = test_table_innerjoin.as_matrix().astype(float)
    # return X_train, y_train, X_val, y_val, X_test
    return X_train, y_train, X_val, y_val, None