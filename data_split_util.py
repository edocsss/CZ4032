import pickle
import os
import pprint
import config
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


"""
This script does data splitting which will be used for training each individual model and the ensemble neural network.
The data split used for training any individual model will be constant to be consistent.

For training the Neural Network ensemble, we use:
Set A = 40% --> training each individual model
Set B = 40% --> the features are given into each individual model. The generated predicted value from each individual model is fed into the ensemble Neural Network as training input
Set C = 20% --> used for testing the ensemble Neural Network model.

For testing the simple averaging ensemble architecture, we use:
Set A = 70% --> for training  each individual model
Set B = 30% --> for testing (given as input to each individual model then do averaging from the output of each individual model)
"""


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

    # Change the splitting ratio according to the experiment you would like to test
    X_A, X_C, Y_A, Y_C = train_test_split(X_matrix, Y_matrix, train_size=0.8)
    X_A, X_B, Y_A, Y_B = train_test_split(X_A, Y_A, train_size=0.5)

    f = open(os.path.join(config.ROOT_DIR, 'data_split.pkl'), 'wb')
    pickle.dump({
        'X_A': X_A,
        'Y_A': Y_A,
        'X_B': X_B,
        'Y_B': Y_B,
        'X_C': X_C,
        'Y_C': Y_C,
        'X_AB': np.concatenate([X_A, X_B]),
        'Y_AB': np.concatenate([Y_A, Y_B])
    }, f)
    f.close()


if __name__ == '__main__':
    build_data_split()