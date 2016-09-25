import pandas as pd
import os
import time
import encoding


DATA_DIR_PATH = '../data'

def load_users_data():
    USERS_DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'users.csv')
    users_df = pd.read_csv(USERS_DATA_FILE_PATH)
    return users_df


def fill_user_age(user):
    age = user['AGE']
    if pd.isnull(age):
        return -1
    else:
        return age


def fill_user_working(user):
    employment = user['WORKING']
    if pd.isnull(employment):
        return 'Not Available'
    else:
        return employment


def fill_user_region(user):
    region = user['REGION']
    if pd.isnull(region):
        return 'Not Available'
    else:
        return region


def clean_list_playback(user, list_type):
    list = user[list_type]

    if not pd.isnull(list):
        list = str(list).lower()
        if '16+' in list or 'more than 16 hours' in list:
            return 17

        elif 'less than an hour' in list:
            return 0

        else:
            return int(list.split(' ')[0])

    else:
        return list


def fill_empty_list_playback(user, list_type, list_mean):
    list = user[list_type]
    if pd.isnull(list):
        return list_mean
    else:
        return list


def fill_empty_question(user, q_col, q_mean):
    q = user[q_col]
    if pd.isnull(q):
        return q_mean
    else:
        return q


def clean_list_own_and_back(users_df):
    users_df['LIST_OWN'] = users_df.apply(clean_list_playback, axis=1, args=('LIST_OWN',))
    users_df['LIST_BACK'] = users_df.apply(clean_list_playback, axis=1, args=('LIST_BACK',))

    users_list_own_mean = users_df['LIST_OWN'].mean()
    users_list_back_mean = users_df['LIST_BACK'].mean()

    users_df['LIST_OWN'] = users_df.apply(fill_empty_list_playback, axis=1, args=('LIST_OWN', users_list_own_mean))
    users_df['LIST_BACK'] = users_df.apply(fill_empty_list_playback, axis=1, args=('LIST_BACK', users_list_back_mean))

    return users_df


def clean_questions(users_df):
    for i in range (1, 20):
        col_name = 'Q' + str(i)
        q_mean = users_df[col_name].mean()
        users_df[col_name] = users_df.apply(fill_empty_question, axis=1, args=(col_name, q_mean))

    return users_df


def encode_user_one_hot(user, use_binary=False):
    gender = user['GENDER'].lower()
    working = user['WORKING'].lower()
    region = user['REGION'].lower()
    music = user['MUSIC'].lower()

    if use_binary:
        gender_encoded = encoding.GENDER_BINARY_MAP[gender]
        working_encoded = encoding.WORKING_BINARY_MAP[working]
        region_encoded = encoding.REGION_BINARY_MAP[region]
        music_encoded = encoding.MUSIC_BINARY_MAP[music]
    else:
        gender_encoded = encoding.GENDER_ONEHOT_MAP[gender]
        working_encoded = encoding.WORKING_ONEHOT_MAP[working]
        region_encoded = encoding.REGION_ONEHOT_MAP[region]
        music_encoded = encoding.MUSIC_ONEHOT_MAP[music]

    for i in range(0, len(gender_encoded)):
        user['GENDER_' + str(i)] = gender_encoded[i]

    for j in range(0, len(working_encoded)):
        user['WORKING_' + str(j)] = working_encoded[j]

    for k in range(0, len(region_encoded)):
        user['REGION_' + str(k)] = region_encoded[k]

    for l in range(0, len(music_encoded)):
        user['MUSIC_' + str(l)] = music_encoded[l]

    return user

def classify_age(user):
    age = int(user['AGE'])

    user['AGE_RANGE_0-15'] = 0
    user['AGE_RANGE_16-25'] = 0
    user['AGE_RANGE_26-35'] = 0
    user['AGE_RANGE_36-45'] = 0
    user['AGE_RANGE_46-55'] = 0
    user['AGE_RANGE_56-65'] = 0
    user['AGE_RANGE_66-'] = 0

    # Ignore ageless user
    if 0 <= age <= 15:
        user['AGE_RANGE_0-15'] = 1
    elif age <= 25:
        user['AGE_RANGE_16-25'] = 1
    elif age <= 35:
        user['AGE_RANGE_26-35'] = 1
    elif age <= 45:
        user['AGE_RANGE_36-45'] = 1
    elif age <= 55:
        user['AGE_RANGE_46-55'] = 1
    elif age <= 65:
        user['AGE_RANGE_56-65'] = 1
    else:
        user['AGE_RANGE_66-'] = 1

    return user


def write_users_df_to_csv(users_df):
    USERS_CLEANED_DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'users_cleaned_binary.csv')
    users_df.to_csv(USERS_CLEANED_DATA_FILE_PATH, sep=',', encoding='utf-8')


if __name__ == '__main__':
    print("Loading users..")
    users_df = load_users_data()

    print("Cleaning users age..")
    users_df['AGE'] = users_df.apply(fill_user_age, axis=1)

    print("Cleaning working..")
    users_df['WORKING'] = users_df.apply(fill_user_working, axis=1)

    print("Cleaning region..")
    users_df['REGION'] = users_df.apply(fill_user_region, axis=1)

    print("Cleaning list own and list back..")
    users_df = clean_list_own_and_back(users_df)

    print("Cleaning questions..")
    users_df = clean_questions(users_df)

    print("Encoding non integer fields..")
    users_df = users_df.apply(encode_user_one_hot, axis=1, args=(True,))

    print('Classifying age...')
    users_df = users_df.apply(classify_age, axis=1)

    print("Writing cleaned data to CSV file..")
    write_users_df_to_csv(users_df)