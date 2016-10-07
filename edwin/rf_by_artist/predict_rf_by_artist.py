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


def predict_rating_by_artist(x, artist_id):
    file_path = os.path.join('models', 'rf_by_artist_model_{}.zip'.format(int(artist_id)))
    f = gzip.GzipFile(file_path, 'rb')
    model = pickle.load(f)
    f.close()

    print('Predicting artist_id = {}..'.format(artist_id))
    predictions = model.predict(x)

    return predictions


def predict_ratings_and_calculate_mse(X, Y):
    full_data = data_util.read_full_data_pickle()
    training_mean_std_per_column = full_data['training_mean_std_per_column']

    X_test_df = build_df_from_test_input(X)
    Y_test_df = pd.DataFrame(data=Y, columns=['Rating'])
    preprocessed_X_test_df = data_util.combine_testing_data(X_test_df, training_mean_std_per_column)

    predictions = []
    grouped_df = group_data_by_artist(preprocessed_X_test_df, Y_test_df)

    for group in grouped_df:
        artist_id = group[0]
        x = group[2].values
        y = group[1].values

        predictions.append((y, predict_rating_by_artist(x, artist_id)))

    calculate_mse_from_predictions(predictions)
    return predictions


def build_df_from_test_input(test_input):
    test_df = pd.DataFrame(data=test_input, columns=['Artist', 'Track', 'User', 'Time'])
    return test_df


def group_data_by_artist(X, Y):
    ARTIST_ID = [x for x in range(50)]
    df = X.join(Y)
    grouped_df_list = []

    for artist_id in ARTIST_ID:
        df_by_artist = df.loc[df['Artist'] == artist_id]
        if len(df_by_artist) > 0:
            grouped_df_list.append((artist_id, df_by_artist['Rating'], df_by_artist.drop('Rating', axis=1)))

    return grouped_df_list


def calculate_mse_from_predictions(prediction_result):
    diff_total = 0
    total_data = 0

    for i in range(len(prediction_result)):
        y_true = prediction_result[i][0]
        y_pred = prediction_result[i][1]
        total_data += len(y_true)

        for j in range(len(y_true)):
            diff_total += math.pow(y_true[j] - y_pred[j], 2)
            print('Predicted value: {}, Actual value: {}'.format(y_pred[j], y_true[j]))

    mse = diff_total / total_data
    print('MSE: {}, RMSE: {}'.format(mse, math.sqrt(mse)))


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = read_data_set()
    predictions = predict_ratings_and_calculate_mse(X_test, Y_test)

    # training_predictions = predict_ratings_and_calculate_mse(X_train, Y_train)
    # result = []
    # for i in range(len(training_predictions)):
    #     result.append((training_predictions[i], Y_train[i]))
    #
    # f = gzip.GzipFile('rf_by_artist_training_predictions_result.zip', 'wb')
    # pickle.dump(result, f)
    # f.close()