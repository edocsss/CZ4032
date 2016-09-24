from edwin import data_util
from sklearn.svm import SVR
from sklearn import cross_validation
import threading


KERNELS = ['rbf', 'poly', 'sigmoid']


def train_svr(X, Y, kernel='rbf'):
    print('SVR Training starting with kernel = {}'.format(kernel))
    svr = SVR(kernel=kernel, verbose=True)

    return cross_validation.cross_val_score(
        svr,
        X,
        Y,
        cv=5,
        scoring='mean_squared_error',
        verbose=10
    )


def run_svr_training(use_binary, file_name):
    X_matrix, Y_matrix = data_util.get_X_and_Y_matrices(use_binary)
    f = open('results/' + file_name, 'w')

    for kernel in KERNELS:
        f.write('SVR Training with kernel = {}\n'.format(kernel))
        scores = train_svr(X_matrix, Y_matrix, kernel)
        [f.write('{}\n'.format(s)) for s in scores]
        f.write('MSE {} (+/- {})'.format(scores.mean(), scores.std() * 2))

        print(scores)

    f.close()


def run_svr_training_binary():
    run_svr_training(True, 'svr_results_binary.txt')


def run_svr_training_onehot():
    run_svr_training(False, 'svr_results_onehot.txt')


if __name__ == '__main__':
    # t_binary = threading.Thread(target=run_svr_training_binary)
    # t_onehot = threading.Thread(target=run_svr_training_onehot)
    #
    # t_binary.start()
    # t_onehot.start()
    #
    # t_binary.join()
    # t_onehot.join()

    run_svr_training_binary()