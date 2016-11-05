import os
import pickle
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from edwin.util import data_util


"""
Run Random Forest model training using the pre-processed training data that can be read using the
<root_dir>/edwin/util/data_util.py script.

This training script does not know how many training data should be used. It only knows that the training data it should
use can be read using <root_dir>/edwin/util/data_util.py.

The number of Random Forest estimators can be changed using the N_ESTIMATORS constant below.
The Random Forest model here is trained using the full data features.
"""


N_ESTIMATORS = 100


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
    file_path = os.path.join('models', 'rf_full_model.p')
    f = open(file_path, 'wb')
    pickle.dump(rf_model, f)
    f.close()


def run_rf_training():
    data = data_util.read_full_data_pickle()
    X_train = data['X_train']
    Y_train = data['Y_train']

    rf_model = train_rf(X_train, Y_train, N_ESTIMATORS)
    store_rf_model(rf_model)
    print('RF Model Full has been trained with 100 n_estimators!')


if __name__ == '__main__':
    run_rf_training()