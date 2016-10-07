import os
import pickle
import pandas as pd
import numpy as np
import config
import data_split_util
import gzip


def read_users_cleaned_data():
    file_path = os.path.join(config.EDWIN_DATA_DIR, 'users_cleaned_binary.csv')
    return pd.read_csv(file_path)


def read_words_cleaned_data():
    file_path = os.path.join(config.EDWIN_DATA_DIR, 'words_cleaned_binary.csv')
    return pd.read_csv(file_path)


def read_train_data():
    file_path = os.path.join(config.EDWIN_DATA_DIR, 'train.csv')
    return pd.read_csv(file_path)


def read_full_data_pickle():
    file_path = os.path.join(config.EDWIN_DATA_DIR, 'full_data.zip')
    f = gzip.GzipFile(file_path, 'rb')
    data = pickle.load(f)
    f.close()

    return data


def read_and_combine_training_data():
    users_df = read_users_cleaned_data()
    words_df = read_words_cleaned_data()
    train_data_split_df = get_train_data_split_df()

    result_df = train_data_split_df.merge(users_df, left_on='User', right_on='RESPID', how='left')
    result_df = result_df.merge(words_df, left_on=['Artist', 'User'], right_on=['Artist', 'User'], how='left')
    result_df = drop_irrelevant_cols_from_merged_df(result_df)

    # Filling in the blanks
    mean_std_per_column = get_mean_std_per_X_column(result_df.drop('Rating', axis=1))
    rating_mean = result_df['Rating'].mean()
    result_df = result_df.apply(fill_in_blank_columns, axis=1, args=(mean_std_per_column, rating_mean))

    result_df = result_df.iloc[np.random.permutation(len(result_df))]
    return result_df, mean_std_per_column


def combine_testing_data(test_split_df, training_mean_std_per_column):
    users_df = read_users_cleaned_data()
    words_df = read_words_cleaned_data()

    result_df = test_split_df.merge(users_df, left_on='User', right_on='RESPID', how='left')
    result_df = result_df.merge(words_df, left_on=['Artist', 'User'], right_on=['Artist', 'User'], how='left')
    result_df = drop_irrelevant_cols_from_merged_df(result_df)
    result_df = result_df.apply(fill_in_blank_columns, axis=1, args=(training_mean_std_per_column,))

    return result_df


def get_train_data_split_df():
    data_split = data_split_util.read_data_split()
    X_train_data_split = data_split['X_train']
    Y_train_data_split = data_split['y_train']

    X_train_data_split_df = pd.DataFrame(data=X_train_data_split, columns=['Artist', 'Track', 'User', 'Time'])
    Y_train_data_split_df = pd.DataFrame(data=Y_train_data_split, columns=['Rating'])

    train_data_split_df = X_train_data_split_df.join(Y_train_data_split_df)
    return train_data_split_df


def drop_irrelevant_cols_from_merged_df(joined_df):
    joined_df = joined_df.drop('User', axis=1)
    joined_df = joined_df.drop('Unnamed: 0_x', axis=1)
    joined_df = joined_df.drop('RESPID', axis=1)
    joined_df = joined_df.drop('Unnamed: 0_y', axis=1)
    joined_df = joined_df.drop('GENDER', axis=1)
    joined_df = joined_df.drop('WORKING', axis=1)
    joined_df = joined_df.drop('REGION', axis=1)
    joined_df = joined_df.drop('MUSIC', axis=1)
    joined_df = joined_df.drop('HEARD_OF', axis=1)
    joined_df = joined_df.drop('OWN_ARTIST_MUSIC', axis=1)

    return joined_df


def fill_in_blank_columns(row, mean_std_per_column, Y_train_mean=None):
    for k, v in mean_std_per_column.items():
        if pd.isnull(row[k]):
            row[k] = v['mean']

    if Y_train_mean is not None and pd.isnull(row['Rating']):
        row['Rating'] = Y_train_mean

    return row


def get_X_and_Y_matrices():
    training_df, training_mean_std_per_column = read_and_combine_training_data()
    Y_df = training_df['Rating']
    X_df = training_df.drop('Rating', axis=1)

    return X_df, Y_df, training_mean_std_per_column


def get_mean_std_per_X_column(X_df):
    result = {}
    for c in X_df.columns:
        result[c] = {}
        result[c]['mean'] = X_df[c].mean()
        result[c]['std'] = X_df[c].std()

    return result


if __name__ == '__main__':
    X_train, Y_train, training_mean_std_per_column = get_X_and_Y_matrices()
    data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'training_mean_std_per_column': training_mean_std_per_column
    }

    f = gzip.GzipFile(os.path.join(config.EDWIN_DATA_DIR, 'full_data.zip'), 'wb')
    pickle.dump(data, f)
    f.close()