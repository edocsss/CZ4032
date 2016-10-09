import pickle
import os
import pprint
import config
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


def read_data_split():
    file_path = os.path.join(config.ROOT_DIR, 'data_split.pkl')
    f = open(file_path, 'rb')
    data_split = pickle.load(f)
    f.close()

    return data_split


def build_data_split():
    file_path = os.path.join(config.ROOT_DIR, 'data', 'train.csv')
    train_df = pd.read_csv(file_path)

    X_matrix = train_df.drop('Rating', axis=1).values.astype(float)
    Y_matrix = train_df['Rating'].values.astype(float)

    X_A, X_B, Y_A, Y_B = train_test_split(X_matrix, Y_matrix, train_size=0.6)
    X_B, X_C, Y_B, Y_C = train_test_split(X_B, Y_B, train_size=0.2)

    X_AB = np.concatenate([X_A, X_B])
    Y_AB = np.concatenate([Y_A, Y_B])

    # A is for training individual model
    # B is for testing individual model
    # AB is to be split further when training NN ensemble (input = Y_predicted per individual model, expected = Y_AB)
    # Train individual model and NN ensemble using X_AB and Y_AB
    # Test the whole end-to-end ensemble model using only X_C and Y_C --> get final error for reporting
    f = open(os.path.join(config.ROOT_DIR, 'data_split.pkl'), 'wb')
    pickle.dump({
        'X_A': X_A,
        'Y_A': Y_A,
        'X_B': X_B,
        'Y_B': Y_B,
        'X_C': X_C,
        'Y_C': Y_C,
        'X_AB': X_AB,
        'Y_AB': Y_AB
    }, f)
    f.close()


if __name__ == '__main__':
    # build_data_split()
    read_data_split()