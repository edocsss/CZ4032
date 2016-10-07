import os
import pickle
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from edwin.util import data_util


def train_rf(X, Y, n=100):
    print('RF Training with n_estimators = {}'.format(n))
    rf_model = RandomForestRegressor(n_estimators=n, verbose=3, n_jobs=-1)
    rf_model.fit(X, Y)
    return rf_model


def cv_rf(X, Y, n=100):
    rf_model = RandomForestRegressor(n_estimators=n, verbose=3, max_features='sqrt')
    scores = cross_val_score(rf_model, X, Y, cv=5, scoring='mean_squared_error', n_jobs=-1, verbose=3)
    print(scores)


def store_rf_model(rf_model):
    file_path = os.path.join('models', 'rf_full_model_ignored.p')
    f = open(file_path, 'wb')
    pickle.dump(rf_model, f)
    f.close()


def run_rf_training():
    data = data_util.read_full_data_pickle()
    X_train = data['X_train']
    Y_train = data['Y_train']

    rf_model = train_rf(X_train, Y_train)
    store_rf_model(rf_model)
    print('RF Model Full has been trained with 100 n_estimators!')


def run_rf_cv():
    data = data_util.read_full_data_pickle()
    X_train = data['X_train']
    Y_train = data['Y_train']
    cv_rf(X_train, Y_train)


if __name__ == '__main__':
    run_rf_training()
    # run_rf_cv()