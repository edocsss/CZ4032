import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
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
X_train_dirty = shared_data['X_train']
X_test_dirty = shared_data['X_test']
y_train_dirty = shared_data['y_train'].astype('float32')
y_test_dirty = shared_data['y_test'].astype('float32')

def augment(x_np, y_np):
    train_features = pd.DataFrame(data=x_np, columns=['Artist', 'Track', 'User', 'Time'])
    train_target = pd.DataFrame(data=y_np, columns=['Rating'])
    train = train_features.join(train_target) # meaning "train.csv"

    users = pd.read_csv(USERS_PATH)
    words = pd.read_csv(WORDS_PATH)

    train_and_users = pd.merge(train, users, left_on="User", right_on="RESPID")
    train_and_users_and_words = pd.merge(train_and_users, words, left_on=["Artist", "User"], right_on=["Artist", "User"])
    return train_and_users_and_words

def get_full_data(x_np, y_np):
    train_aug = augment(x_np, y_np)

    cols = [col for col in train_aug.columns if col not in ["Rating", "HEARD_OF", "OWN_ARTIST_MUSIC", "GENDER", "WORKING", "REGION", "MUSIC", 'Unnamed: 0_x', 'Unnamed: 0_y']]

    new_x_np = (train_aug[cols].values).astype('float32')
    new_y_np = (train_aug[["Rating"]].values).astype('float32')
    new_y_np = np.ravel(new_y_np)

    return new_x_np, new_y_np

X_train, y_train = get_full_data(X_train_dirty, y_train_dirty)
X_test, y_test = get_full_data(X_test_dirty, y_test_dirty)

# Now this~
params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 1,
          'learning_rate': 0.5, 'loss': 'ls'}
# clf = ensemble.GradientBoostingRegressor(**params)
# model = clf.fit(X_train, y_train)

# Who likes pickle?
# f = open('gbr_ne500_maxdepth5_minsamplessplit1_lr0.5_lossls.pkl', 'wb')
# pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
f = open('gbr_ne500_maxdepth5_minsamplessplit1_lr0.5_lossls.pkl', 'rb')
model = pickle.load(f)
f.close()

preds = model.predict(X_train)
res = []
for i in range(len(preds)):
    res.append((preds[i], y_train[i]))

fp = open('gbr_preds.pkl', 'wb')
pickle.dump(res, fp, protocol=pickle.HIGHEST_PROTOCOL)
fp.close()

# Check loss
train_error = mean_squared_error(y_train, model.predict(X_train))
test_error = mean_squared_error(y_test, model.predict(X_test))

print("GradientBoostingRegressor")
print("Parameters: %s" % json.dumps(params))
print ("training data MSE")
print(train_error)
print ("test data MSE")
print(test_error)
