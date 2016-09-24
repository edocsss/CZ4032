from edwin import data_util
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


KERNELS = ['rbf', 'poly', 'sigmoid']
tuned_parameters = [
    {
        'kernel': ['rbf'],
        'gamma': [1e-3, 1e-4],
        'C': [1, 10, 100, 1000]
    },
    {
        'kernel': ['linear'],
        'C': [1, 10, 100, 1000]
    }
]


def run_svr_training(use_binary, file_name):
    X_matrix, Y_matrix = data_util.get_X_and_Y_matrices(use_binary)
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X_matrix, Y_matrix, test_size=0.7)

    f = open('results/' + file_name, 'w')

    clf = GridSearchCV(
        SVR(C=1),
        tuned_parameters,
        cv=3,
        scoring='mean_squared_error',
        verbose=10
    )

    clf.fit(X_train, Y_train)
    print('Best parameters set found on the testing set:')
    f.write('Best parameters set found on the testing set:\n')
    print(clf.best_params_)
    f.write(clf.best_params_)
    f.write('\n\n')

    print('Grid scores:')
    f.write('Grid scores:\n')

    for params, mean_score, scores in clf.grid_scores_:
        print('{} (+/- {}) for {}'.format(mean_score, scores.std() * 2, params))
        f.write('{} (+/- {}) for {}\n\n'.format(mean_score, scores.std() * 2, params))

    Y_true, Y_pred = Y_test, clf.predict(X_test)
    print(classification_report(Y_true, Y_pred))
    f.write(classification_report(Y_true, Y_pred))
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