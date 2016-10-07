import os
import pickle
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from edwin.util import data_util
import gzip


def train_rf_by_artist(X, Y, n=100):
    print('RF Training with n_estimators = {}'.format(n))
    rf_model = RandomForestRegressor(n_estimators=n, n_jobs=-1, verbose=0)
    rf_model.fit(X, Y)
    return rf_model


def cv_rf_by_artist(X, Y, n=50):
    rf_model = RandomForestRegressor(n_estimators=n, verbose=3, max_features='sqrt')
    scores = cross_val_score(rf_model, X, Y, cv=5, scoring='mean_squared_error', n_jobs=-1, verbose=3)
    print(scores)


def store_rf_by_artist_model(rf_model, artist_id):
    file_path = os.path.join('models', 'rf_by_artist_model_{}.zip'.format(artist_id))
    f = gzip.GzipFile(file_path, 'wb')
    pickle.dump(rf_model, f)
    f.close()


def group_training_data_by_artist(X_train, Y_train):
    ARTIST_ID = [x for x in range(0, 50)]
    training_df = X_train.join(Y_train)
    grouped_df_list = []

    for artist_id in ARTIST_ID:
        df_by_artist = training_df.loc[training_df['Artist'] == artist_id]
        grouped_df_list.append((df_by_artist['Rating'], df_by_artist.drop('Rating', axis=1)))

    return grouped_df_list


def run_rf_by_artist_training():
    data = data_util.read_full_data_pickle()
    X_train = data['X_train']
    Y_train = data['Y_train']

    training_data_by_artist = group_training_data_by_artist(X_train, Y_train)
    for i, training_df in enumerate(training_data_by_artist):
        Y = training_df[0]
        X = training_df[1]

        rf_model = train_rf_by_artist(X, Y)
        store_rf_by_artist_model(rf_model, i)
        print('RF Model by Artist {} has been trained with 100 n_estimators!'.format(i))


def run_rf_by_artist_cv():
    data = data_util.read_full_data_pickle()
    X_train = data['X_train']
    Y_train = data['Y_train']

    training_data_by_artist = group_training_data_by_artist(X_train, Y_train)
    for training_df in training_data_by_artist:
        Y = training_df[0]
        X = training_df[1]
        cv_rf_by_artist(X, Y, n=100)


if __name__ == '__main__':
    run_rf_by_artist_training()
    # run_rf_by_artist_cv()