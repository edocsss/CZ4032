import math
import os
import pickle
import pandas as pd
import data_split_util
import gzip
from edwin.util import data_util


"""
Predict the rating for a given input using the Random Forest model trained using
<root_dir>/edwin/rf_by_artist/train_rf_by_artist.py.

The data to be predicted needs to be pre-processed using the following procedures that we use for pre-processing the
training data for building the Random Forest. The pre-processing is done by the <root_dir>/edwin/util/data_util.py script.

In addition, for each testing data, we need to know the Artist ID of that particular data. This is because we have one
Random Forest model for each Artist ID. Testing data with Artist ID = X will have its rating predicted by Random Forest model(X).

After predicting the rating for a set of testing data, the predictions are stored in a zipped pickle file.
This is needed since we are going to use the predicted ratings to train the ensemble Neural Network or do the simple averaging
ensemble method.
"""



def read_data_set():
    data_split = data_split_util.read_data_split()
    return data_split['X_A'], data_split['Y_A'], data_split['X_B'], data_split['Y_B'], data_split['X_C'], data_split['Y_C']


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

    test_df = build_df_from_test_input(X, Y)
    preprocessed_test_df = data_util.combine_testing_data(test_df, training_mean_std_per_column)

    predictions = []
    grouped_df = group_data_by_artist(preprocessed_test_df.drop('Rating', axis=1), preprocessed_test_df['Rating'])

    for group in grouped_df:
        artist_id = group[0]
        y = group[1].values
        x = group[2].values
        indices = group[3].values

        prediction_result = predict_rating_by_artist(x, artist_id)
        for i in range(len(prediction_result)):
            predictions.append((indices[i], prediction_result[i], y[i]))

    predictions.sort(key=lambda item: item[0])
    predictions = [(p[1], p[2]) for p in predictions]
    calculate_mse_from_predictions(predictions)
    return predictions


def build_df_from_test_input(X, Y):
    X_test_df = pd.DataFrame(data=X, columns=['Artist', 'Track', 'User', 'Time'])
    Y_test_df = pd.DataFrame(data=Y, columns=['Rating'])
    test_df = X_test_df.join(Y_test_df)
    return test_df


def group_data_by_artist(X, Y):
    ARTIST_ID = [x for x in range(50)]
    df = X.join(Y)
    grouped_df_list = []

    for artist_id in ARTIST_ID:
        df_by_artist = df.loc[df['Artist'] == artist_id]
        if len(df_by_artist) > 0:
            grouped_df_list.append((artist_id, df_by_artist['Rating'], df_by_artist.drop('Rating', axis=1), df_by_artist.index))

    return grouped_df_list


def calculate_mse_from_predictions(prediction_result):
    diff_total = 0
    for i in range(len(prediction_result)):
        y_true = prediction_result[i][1]
        y_pred = prediction_result[i][0]
        diff_total += math.pow(y_true - y_pred, 2)

    mse = diff_total / len(prediction_result)
    print('MSE: {}, RMSE: {}'.format(mse, math.sqrt(mse)))


if __name__ == '__main__':
    # Change the dataset to be predicted accordingly
    X_A, Y_A, X_B, Y_B, X_C, Y_C = read_data_set()
    predictions = predict_ratings_and_calculate_mse(X_C, Y_C)

    f = gzip.GzipFile('rf_by_artist_training_predictions_result_C.zip', 'wb')
    pickle.dump(predictions, f)
    f.close()