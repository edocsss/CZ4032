import os
import sys
import time
import math
import random
import pickle
import numpy as np
import pandas as pd
from helper import *
from sklearn.ensemble import RandomForestRegressor as RFG

# Helper function
def split(d, val_ratio=.3, shuffle=True):
    m = d.shape[0]
    if shuffle:
        shuffled_idx = np.arange(m)
        np.random.shuffle(shuffled_idx)
        d = d[shuffled_idx]
    train_ratio = 1. - val_ratio
    assert train_ratio > 0., 'wtf?'
    train_size = int(math.ceil(train_ratio * m))
    val_size = m - train_size
    X_train = d[:train_size, 1:]
    y_train = d[:train_size, [0]]
    X_val = d[train_size:, 1:]
    y_val = d[train_size:, [0]]
    return X_train, y_train, X_val, y_val

if __name__ == '__main__':
    # RandomForest by User and their music habit
    # Loading dataset
    d, t = load_dataset(debug=True, suffix='onehot', suffix2='A')
    # Filters for obtaining only important columns/fields
    col_filters = ['LIKE_ARTIST', 'Uninspired', 'Sophisticated', 'Aggressive',
        'Edgy', 'Sociable', 'Laid back', 'Wholesome', 'Uplifting', 'Intriguing',
        'Legendary', 'Free', 'Thoughtful', 'Outspoken', 'Serious',
        'Good lyrics', 'Unattractive', 'Confident', 'Old', 'Youthful', 'Boring',
        'Current', 'Colourful', 'Stylish', 'Cheap', 'Irrelevant', 'Heartfelt',
        'Calm', 'Pioneer', 'Outgoing', 'Inspiring', 'Beautiful', 'Fun',
        'Authentic', 'Credible', 'Way out', 'Cool', 'Catchy', 'Sensitive',
        'Mainstream', 'Superficial', 'Annoying', 'Dark', 'Passionate',
        'Not authentic', 'Background', 'Timeless', 'Depressing', 'Original',
        'Talented', 'Worldly', 'Distinctive', 'Approachable', 'Genius',
        'Trendsetter', 'Noisy', 'Upbeat', 'Relatable', 'Energetic', 'Exciting',
        'Emotional', 'Nostalgic', 'None of these', 'Progressive', 'Sexy',
        'Over', 'Rebellious', 'Fake', 'Cheesy', 'Popular', 'Superstar',
        'Relaxed', 'Intrusive', 'Unoriginal', 'Dated', 'Iconic',
        'Unapproachable', 'Classic', 'Playful', 'Arrogant', 'Warm', 'Soulful',
        'HEARD_OF_0', 'HEARD_OF_1', 'HEARD_OF_2', 'HEARD_OF_3', 'HEARD_OF_4',
        'HEARD_OF_5', 'HEARD_OF_6', 'OWN_ARTIST_MUSIC_0', 'OWN_ARTIST_MUSIC_1',
        'OWN_ARTIST_MUSIC_2', 'OWN_ARTIST_MUSIC_3', 'OWN_ARTIST_MUSIC_4',
        'LIST_OWN', 'LIST_BACK', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7',
        'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18']
    train_df = d.loc[:, ['Rating'] + col_filters]
    # test_df = t.loc[:, col_filters]
    dataset = train_df.as_matrix().astype(float)
    X_train, y_train, X_val, y_val = split(dataset, val_ratio=0., shuffle=False)
    # For fitting into RF
    y_train = y_train.flatten()
    y_val = y_val.flatten()
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    # X_test = test_df.as_matrix().astype(float)

    print('Saving RAM')
    # For the love of my laptop, please save the RAM
    del train_df
    # del test_df
    print('RAM saved!')

    # Preparing to train
    print('Dataset loaded! Training with RFG...')
    rf = RFG(n_estimators=38)
    rf.fit(X_train, y_train)

    # Saving model to pickle file
    with open('rf1.pkl', 'wb') as f:
        pickle.dump(rf,f)
    print('Training done!')
    # y_hat=rf.predict(X_train)
    # with open('rf1_predicts.pkl', 'wb') as f:
    #     pickle.dump(np.append(y_hat[:,None], y_train[:,None], axis=1), f)

    # Predicting
    X_B, y_B = wrapper(suffix='B', filters=col_filters)
    y_B_hat = rf.predict(X_B)
    with open('rf1_y_B_hat.pkl', 'wb') as f:
        pickle.dump(np.append(y_B_hat[:,None], y_B, axis=1), f)
