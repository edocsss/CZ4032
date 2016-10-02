import os
import sys
import math
import pandas
import numpy as np

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