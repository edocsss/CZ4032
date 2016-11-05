import math
import os
import pickle
import pandas as pd
import data_split_util
import gzip
import numpy as np
from edwin.util import data_util
from sklearn.metrics import mean_squared_error


"""
Predict the rating for a given input using the Random Forest model trained using
<root_dir>/edwin/rf_full/train_rf_full.py.

The data to be predicted needs to be pre-processed using the following procedures that we use for pre-processing the
training data for building the Random Forest. The pre-processing is done by the <root_dir>/edwin/util/data_util.py script

After predicting the rating for a set of testing data, the predictions are stored in a zipped pickle file.
This is needed since we are going to use the predicted ratings to train the ensemble Neural Network or do the simple averaging
ensemble method.
"""


def read_data_set():
    data_split = data_split_util.read_data_split()
    return data_split['X_A'], data_split['Y_A'], data_split['X_B'], data_split['Y_B'], data_split['X_C'], data_split['Y_C']


def build_df_from_test_input(X, Y):
    X_test_df = pd.DataFrame(data=X, columns=['Artist', 'Track', 'User', 'Time'])
    Y_test_df = pd.DataFrame(data=Y, columns=['Rating'])
    test_df = X_test_df.join(Y_test_df)
    return test_df


def predict_rating(X):
    file_path = os.path.join('models', 'rf_full_model.p')
    f = open(file_path, 'rb')
    model = pickle.load(f)
    f.close()

    return model.predict(X)


def predict_rf_full(X_test_matrix, Y_test_matrix):
    full_data = data_util.read_full_data_pickle()
    training_mean_std_per_column = full_data['training_mean_std_per_column']

    test_df = build_df_from_test_input(X_test_matrix, Y_test_matrix)
    preprocessed_test_df = data_util.combine_testing_data(test_df, training_mean_std_per_column)

    predictions = predict_rating(preprocessed_test_df.drop('Rating', axis=1))
    calculate_mse_from_predictions(predictions, preprocessed_test_df['Rating'])
    return predictions, preprocessed_test_df.drop('Rating', axis=1), preprocessed_test_df['Rating']


def calculate_mse_from_predictions(predictions, Y_true):
    mse = mean_squared_error(Y_true, predictions)
    print('MSE: {}, RMSE: {}'.format(mse, math.sqrt(mse)))


if __name__ == '__main__':
    # Change the dataset to be predicted accordingly
    X_A, Y_A, X_B, Y_B, X_C, Y_C = read_data_set()
    predictions, X_C, Y_C = predict_rf_full(X_C, Y_C)

    result = []
    for i in range(len(predictions)):
        result.append((predictions[i], Y_C[i]))

    # Store the prediction for future ensemble NN training
    f = gzip.GzipFile('rf_full_training_predictions_result_C_NN.zip', 'wb')
    pickle.dump(result, f)
    f.close()