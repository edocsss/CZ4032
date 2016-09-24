from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from edwin import data_util
import threading


N_RF_TREES = [1, 2, 5, 10, 20, 30, 50, 100]


def train_rf(X, Y, n=100):
    print('RF Training with n_estimators = {}'.format(n))
    rf_model = RandomForestRegressor(n_estimators=n)
    scores = cross_validation.cross_val_score(
        rf_model,
        X,
        Y,
        cv=5,
        scoring='mean_squared_error',
        verbose=10
    )

    return scores


def run_rf_training(use_binary, file_name):
    X_matrix, Y_matrix = data_util.get_X_and_Y_matrices(use_binary)
    f = open('results/' + file_name, 'w')

    for n in N_RF_TREES:
        print('Training RF with binary encoding, n = {}'.format(n))
        f.write('Training RF with binary encoding, n = {}\n'.format(n))

        scores = train_rf(X_matrix, Y_matrix, n)
        [f.write('{}\n'.format(s)) for s in scores]
        f.write('MSE {} (+/- {})'.format(scores.mean(), scores.std() * 2))

        print(scores)
        print('MSE {} (+/- {})'.format(scores.mean(), scores.std() * 2))

    f.close()


def run_rf_training_binary():
    run_rf_training(True, 'rf_results_binary.txt')


def run_rf_training_onehot():
    run_rf_training(False, 'rf_results_onehot.txt')


if __name__ == '__main__':
    t_binary = threading.Thread(target=run_rf_training_binary)
    t_onehot = threading.Thread(target=run_rf_training_onehot)

    t_binary.start()
    t_onehot.start()

    t_binary.join()
    t_onehot.join()