import pickle
import os
import pprint
import config
import pandas as pd
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

    X_train, X_test, Y_train, Y_test = train_test_split(X_matrix, Y_matrix, train_size=0.7)
    f = open(os.path.join(config.ROOT_DIR, 'data_split.pkl'), 'wb')
    pickle.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': Y_train,
        'y_test': Y_test
    }, f)
    f.close()


if __name__ == '__main__':
    build_data_split()