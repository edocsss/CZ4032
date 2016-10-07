import math
import os
import pickle
import pandas as pd
import data_split_util
import gzip
from edwin.util import data_util


def read_data_set():
    data_split = data_split_util.read_data_split()
    return data_split['X_train'], data_split['y_train'], data_split['X_test'], data_split['y_test']


def build_df_from_test_input(test_input):
    test_df = pd.DataFrame(data=test_input, columns=['Artist', 'Track', 'User', 'Time'])
    return test_df


def predict_rating(X):
    file_path = os.path.join('models', 'rf_full_model_ignored.p')
    f = open(file_path, 'rb')
    model = pickle.load(f)
    f.close()

    return model.predict(X)


def predict_rf_full(X_test_matrix, Y_test_matrix):
    full_data = data_util.read_full_data_pickle()
    training_mean_std_per_column = full_data['training_mean_std_per_column']

    X_test_df = build_df_from_test_input(X_test_matrix)
    preprocessed_X_test_df = data_util.combine_testing_data(X_test_df, training_mean_std_per_column)

    predictions = predict_rating(preprocessed_X_test_df.values)
    calculate_mse_from_predictions(predictions, Y_test_matrix)
    return predictions


def calculate_mse_from_predictions(predictions, Y_true):
    diff_total = 0
    for i in range(len(predictions)):
        diff_total += math.pow((predictions[i] - Y_true[i]), 2)
        print('Predicted value: {}, Actual value: {}'.format(predictions[i], Y_true[i]))

    mse = diff_total / len(Y_true)
    print('MSE: {}, RMSE: {}'.format(mse, math.sqrt(mse)))


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = read_data_set()
    predictions = predict_rf_full(X_test, Y_test)

    # training_predictions = predict_rf_full(X_train, Y_train)
    # result = []
    # for i in range(len(training_predictions)):
    #     result.append((training_predictions[i], Y_train[i]))
    #
    # f = gzip.GzipFile('rf_full_training_predictions_result.zip', 'wb')
    # pickle.dump(result, f)
    # f.close()