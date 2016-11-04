import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
import math


ROOT_PATH = os.path.join("..")
TRAIN_DATA_PATH = os.path.join(ROOT_PATH, 'data_split.pkl')
DATA_PATH = os.path.join("..", "data")
USERS_PATH = os.path.join(DATA_PATH, "users_cleaned_binary.csv")
WORDS_PATH = os.path.join(DATA_PATH, "words_cleaned_binary.csv")

def load_data():
	f = open(TRAIN_DATA_PATH, 'rb')
	data_from_pickle = pickle.load(f)
	X_A = data_from_pickle['X_A']
	X_B = data_from_pickle['X_B']
	X_C = data_from_pickle['X_C']
	X_AB = data_from_pickle['X_AB']
	Y_A = data_from_pickle['Y_A']
	Y_B = data_from_pickle['Y_B']
	Y_C = data_from_pickle['Y_C']
	Y_AB = data_from_pickle['Y_AB']
	f.close()
	return X_A, Y_A, X_B, Y_B, X_C, Y_C, X_AB, Y_AB

def get_additional_features(x, y):
	train_features = pd.DataFrame(data=x, columns=['Artist', 'Track', 'User', 'Time'])
	train_target = pd.DataFrame(data=y, columns=['Rating'])
	train_data = train_features.join(train_target)

	#Prepare data that will be joined (i.e. get additional features)
	users = pd.read_csv(USERS_PATH)
	words = pd.read_csv(WORDS_PATH)

	#Join train and user
	train_join_users = pd.merge(train_data, users, left_on="User", right_on="RESPID")
	#Join train and user, with word
	train_join_users_join_words = pd.merge(train_join_users, words, left_on=["Artist", "User"], right_on=["Artist", "User"])

	cols = [col for col in train_join_users_join_words.columns if col not in ["Rating", "HEARD_OF", "OWN_ARTIST_MUSIC", "GENDER", "WORKING", "REGION", "MUSIC", 'Unnamed: 0_x', 'Unnamed: 0_y']]

	x_result = (train_join_users_join_words[cols].values)
	y_result = (train_join_users_join_words[["Rating"]].values)
	return x_result, y_result

def save_to_pickle(data, filename):
	f = open(filename, 'wb')
	pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
	f.close()

def main():
	#Load
	X_A, Y_A, X_B, Y_B, X_C, Y_C, X_AB, Y_AB = load_data()
	X_A, Y_A = get_additional_features(X_A, Y_A)
	X_B, Y_B = get_additional_features(X_B, Y_B)
	X_C, Y_C = get_additional_features(X_C, Y_C)
	X_AB, Y_AB = get_additional_features(X_AB, Y_AB)

	a = 0.001
	print('Alpha = {}'.format(a))

	model = Ridge(alpha=a)
	model.fit(X_AB, Y_AB)

	predicted_result = model.predict(X_C)
	mse = mean_squared_error(Y_C, predicted_result)
	print('C_MSE: {}, C_RMSE: {}'.format(mse, math.sqrt(mse)))

	result = []
	for i in range(len(predicted_result)):
		result.append((predicted_result[i], Y_C[i]))

	#Save data
	save_to_pickle(model, 'ridge_model_C.pkl')
	save_to_pickle(result, 'ridge_prediction_C2.pkl')

main()