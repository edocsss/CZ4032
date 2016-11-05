import os
import pickle
import pandas as pd
import gzip
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from edwin.util import data_util


"""
Run Linear Regression model training using the pre-processed training data that can be read using the
<root_dir>/edwin/util/data_util.py script.

This training script does not know how many training data should be used. It only knows that the training data it should
use can be read using <root_dir>/edwin/util/data_util.py.

The Linear Regression model here is trained using the full data features but ONE MODEL will be generated for each artist ID.
This means that one Linear Regression model is trained using data that corresponds to a particular artist ID.

Thus:
model(i) corresponds to a Linear Regression model trained using data that has the value i for the column Artist ID.
"""


def train_lr_by_artist(X, Y):
    print('LR Training')
    lr_model = LinearRegression(n_jobs=-1, normalize=True)
    lr_model.fit(X, Y)
    return lr_model


def cv_lr_by_artist(X, Y):
    lr_model = LinearRegression(n_jobs=-1, normalize=False)
    scores = cross_val_score(lr_model, X, Y, cv=5, scoring='mean_squared_error', n_jobs=-1)
    print(scores)


def store_lr_by_artist_model(lr_model, artist_id):
    file_path = os.path.join('models', 'lr_by_artist_model_{}.zip'.format(artist_id))
    f = gzip.GzipFile(file_path, 'wb')
    pickle.dump(lr_model, f)
    f.close()


def group_training_data_by_artist(X_train, Y_train):
    ARTIST_ID = [x for x in range(0, 50)]
    training_df = X_train.join(Y_train)
    grouped_df_list = []

    for artist_id in ARTIST_ID:
        df_by_artist = training_df.loc[training_df['Artist'] == artist_id]
        grouped_df_list.append((df_by_artist['Rating'], df_by_artist.drop('Rating', axis=1)))

    return grouped_df_list


def run_lr_by_artist_training():
    data = data_util.read_full_data_pickle()
    X_train = data['X_train']
    Y_train = data['Y_train']

    training_data_by_artist = group_training_data_by_artist(X_train, Y_train)
    for i, training_df in enumerate(training_data_by_artist):
        Y = training_df[0]
        X = training_df[1]

        lr_model = train_lr_by_artist(X, Y)
        store_lr_by_artist_model(lr_model, i)
        print('LR Model by Artist {} has been trained!')


def run_lr_by_artist_cv():
    data = data_util.read_full_data_pickle()
    X_train = data['X_train']
    Y_train = data['Y_train']

    training_data_by_artist = group_training_data_by_artist(X_train, Y_train)
    for training_df in training_data_by_artist:
        Y = training_df[0]
        X = training_df[1]
        cv_lr_by_artist(X, Y)


if __name__ == '__main__':
    run_lr_by_artist_training()
    # run_lr_by_artist_cv()