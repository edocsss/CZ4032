from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
from edwin import data_util
import threading


N_GBR_TREES = [1, 30, 50, 100, 200]


def train_gbr(X, Y, n=100):
    print('GBR Training with n_estimators = {}'.format(n))
    gbr_model = GradientBoostingRegressor(n_estimators=n, verbose=3)
    scores = cross_validation.cross_val_score(
        gbr_model,
        X,
        Y,
        cv=5,
        scoring='mean_squared_error',
        verbose=10,
        n_jobs=3
    )

    return scores


def run_gbr_training(use_binary, file_name):
    X_matrix, Y_matrix = data_util.get_X_and_Y_matrices(use_binary)
    f = open('results/' + file_name, 'w')

    for n in N_GBR_TREES:
        print('Training GTR with binary encoding, n = {}'.format(n))
        f.write('Training GTR with binary encoding, n = {}\n'.format(n))

        scores = train_gbr(X_matrix, Y_matrix, n)
        [f.write('{}\n'.format(s)) for s in scores]
        f.write('MSE {} (+/- {})'.format(scores.mean(), scores.std() * 2))

        print(scores)
        print('MSE {} (+/- {})'.format(scores.mean(), scores.std() * 2))

    f.close()


def run_gbr_training_binary():
    run_gbr_training(True, 'gbr_results_binary.txt')


def run_gbr_training_onehot():
    run_gbr_training(False, 'gbr_results_onehot.txt')


if __name__ == '__main__':
    run_gbr_training_binary()