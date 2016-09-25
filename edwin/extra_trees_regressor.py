from sklearn.ensemble import ExtraTreesRegressor
from sklearn import cross_validation
from edwin import data_util
import threading


N_ETR_TREES = [1, 2, 5, 10, 20, 30, 50, 100]


def train_etr(X, Y, n=100):
    print('ETR Training with n_estimators = {}'.format(n))
    etr_model = ExtraTreesRegressor(n_estimators=n, verbose=3)
    scores = cross_validation.cross_val_score(
        etr_model,
        X,
        Y,
        cv=5,
        scoring='mean_squared_error',
        verbose=10,
        n_jobs=3
    )

    return scores


def run_etr_training(use_binary, file_name):
    X_matrix, Y_matrix = data_util.get_X_and_Y_matrices(use_binary)
    f = open('results/' + file_name, 'w')

    for n in N_ETR_TREES:
        print('Training ETR with binary encoding, n = {}'.format(n))
        f.write('Training ETR with binary encoding, n = {}\n'.format(n))

        scores = train_etr(X_matrix, Y_matrix, n)
        [f.write('{}\n'.format(s)) for s in scores]
        f.write('MSE {} (+/- {})'.format(scores.mean(), scores.std() * 2))

        print(scores)
        print('MSE {} (+/- {})'.format(scores.mean(), scores.std() * 2))

    f.close()


def run_etr_training_binary():
    run_etr_training(True, 'etr_results_binary.txt')


def run_etr_training_onehot():
    run_etr_training(False, 'etr_results_onehot.txt')


if __name__ == '__main__':
    run_etr_training_binary()