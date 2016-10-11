import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.linear_model import LassoLarsCV
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

ROOT_PATH = os.path.join("..")
TRAIN_DATA_PATH = os.path.join(ROOT_PATH, 'data_split.pkl')
DATA_PATH = os.path.join("..", "data")
USERS_PATH = os.path.join(DATA_PATH, "users_cleaned_binary.csv")
WORDS_PATH = os.path.join(DATA_PATH, "words_cleaned_binary.csv")

# Load data
f = open(TRAIN_DATA_PATH, 'rb')
shared_data = pickle.load(f)
f.close()
# X_train_dirty = shared_data['X_A']
# X_val_dirty = shared_data['X_B']
# X_train_and_val_dirty = shared_data['X_AB']
X_test_dirty = shared_data['X_C']
# y_train_dirty = shared_data['Y_A'].astype('float32')
# y_val_dirty = shared_data['Y_B'].astype('float32')
# y_train_and_val_dirty = shared_data['Y_AB'].astype('float32')
y_test_dirty = shared_data['Y_C'].astype('float32')

# copied from Edwin :P
def fill_in_blank_columns(row, mean_std_per_column, Y_train_mean=None):
    for k, v in mean_std_per_column.items():
        if pd.isnull(row[k]):
            row[k] = v['mean']
    if Y_train_mean is not None and pd.isnull(row['Rating']):
        row['Rating'] = Y_train_mean
    return row
def get_mean_std_per_X_column(X_df):
    result = {}
    for c in X_df.columns:
        result[c] = {}
        result[c]['mean'] = X_df[c].mean()
        result[c]['std'] = X_df[c].std()

    return result

def augment(x_np, y_np):
    train_features = pd.DataFrame(data=x_np, columns=['Artist', 'Track', 'User', 'Time'])
    train_target = pd.DataFrame(data=y_np, columns=['Rating'])
    train = train_features.join(train_target) # meaning "train.csv"

    users = pd.read_csv(USERS_PATH)
    words = pd.read_csv(WORDS_PATH)

    train_and_users = pd.merge(train, users, left_on="User", right_on="RESPID", how='left')
    train_and_users_and_words = pd.merge(train_and_users, words, left_on=["Artist", "User"], right_on=["Artist", "User"])

    # drop
    for attr in ["HEARD_OF", "OWN_ARTIST_MUSIC", "GENDER", "WORKING", "REGION", "MUSIC", 'Unnamed: 0_x', 'Unnamed: 0_y']:
        train_and_users_and_words = train_and_users_and_words.drop(attr, axis=1)
    
    mean_std_per_column = get_mean_std_per_X_column(train_and_users_and_words.drop('Rating', axis=1))
    rating_mean = train_and_users_and_words['Rating'].mean()
    train_and_users_and_words = train_and_users_and_words.apply(fill_in_blank_columns, axis=1, args=(mean_std_per_column, rating_mean))


    return train_and_users_and_words

def get_full_data(x_np, y_np):
    train_aug = augment(x_np, y_np)

    cols = [col for col in train_aug.columns if col not in ["Rating", "HEARD_OF", "OWN_ARTIST_MUSIC", "GENDER", "WORKING", "REGION", "MUSIC", 'Unnamed: 0_x', 'Unnamed: 0_y']]

    new_x_np = (train_aug[cols].values).astype('float32')
    new_y_np = (train_aug[["Rating"]].values).astype('float32')
    new_y_np = np.ravel(new_y_np)

    return new_x_np, new_y_np

# X_train, y_train = get_full_data(X_train_dirty, y_train_dirty)
# X_val, y_val = get_full_data(X_val_dirty, y_val_dirty)
# X_train_and_val, y_train_and_val = get_full_data(X_train_and_val_dirty, y_train_and_val_dirty)
X_test, y_test = get_full_data(X_test_dirty, y_test_dirty)


# Who likes pickle?
f = open('gbr_ne500_maxdepth5_minsamplessplit1_lr0.5_lossls3.pkl', 'rb')
model = pickle.load(f)
f.close()

# Check loss
# train_error = mean_squared_error(y_train, model.predict(X_train))
# val_error = mean_squared_error(y_val, model.predict(X_val))
test_error = mean_squared_error(y_test, model.predict(X_test))

print("GradientBoostingRegressor Predict")
# print("training data MSE (A)")
# print(train_error)
# print("validation data MSE (B)")
# print(val_error)
print("test data MSE (C)")
print(test_error)


# Predict for Peter
preds = model.predict(X_test)
res = []
for i in range(len(preds)):
    res.append((preds[i], y_test[i]))

fp = open('gbr_preds_test.pkl', 'wb')
pickle.dump(res, fp, protocol=pickle.HIGHEST_PROTOCOL)
fp.close()
