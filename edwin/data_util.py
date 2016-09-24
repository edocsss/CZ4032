import os
import pandas as pd
import numpy as np


def read_users_cleaned_data(use_binary=False):
    if use_binary:
        file_path = os.path.join('../data/users_cleaned_binary.csv')
    else:
        file_path = os.path.join('../data/users_cleaned_onehot.csv')

    return pd.read_csv(file_path)


def read_words_cleaned_data(use_binary=False):
    if use_binary:
        file_path = os.path.join('../data/words_cleaned_binary.csv')
    else:
        file_path = os.path.join('../data/words_cleaned_onehot.csv')

    return pd.read_csv(file_path)


def read_train_data():
    file_path = os.path.join('../data/train.csv')
    return pd.read_csv(file_path)


def read_and_combine_training_data(use_binary=True):
    users_df = read_users_cleaned_data(use_binary)
    words_df = read_words_cleaned_data(use_binary)
    train_df = read_train_data()

    result_df = train_df.merge(users_df, left_on='User', right_on='RESPID')
    result_df = result_df.merge(words_df, left_on=['Artist', 'User'], right_on=['Artist', 'User'])

    result_df = result_df.drop('User', axis=1)
    result_df = result_df.drop('Unnamed: 0_x', axis=1)
    result_df = result_df.drop('RESPID', axis=1)
    result_df = result_df.drop('Unnamed: 0_y', axis=1)
    result_df = result_df.drop('GENDER', axis=1)
    result_df = result_df.drop('WORKING', axis=1)
    result_df = result_df.drop('REGION', axis=1)
    result_df = result_df.drop('MUSIC', axis=1)
    result_df = result_df.drop('HEARD_OF', axis=1)
    result_df = result_df.drop('OWN_ARTIST_MUSIC', axis=1)

    result_df = result_df.iloc[np.random.permutation(len(result_df))]
    return result_df


def normalize_data(df):
    return (df - df.mean()) / df.std()


def get_X_and_Y_matrices(use_binary=True):
    training_df = read_and_combine_training_data(use_binary)
    Y_df = training_df['Rating']
    X_df = training_df.drop('Rating', axis=1)
    X_df = normalize_data(X_df)

    X_matrix = X_df.values
    Y_matrix = Y_df.values

    return X_matrix, Y_matrix