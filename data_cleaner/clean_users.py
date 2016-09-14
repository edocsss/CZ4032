import pandas as pd
import os
import encoding.users as users_encoding


DATA_DIR_PATH = '../data'


def load_users_data():
    USERS_DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'users.csv')
    users_df = pd.read_csv(USERS_DATA_FILE_PATH)
    return users_df


def fill_user_age(user):
    age = user['AGE']
    if pd.isnull(age):
        return 'NA'
    else:
        return age


def fill_user_working(user):
    employment = user['WORKING']
    if pd.isnull(employment):
        return 'NA'
    else:
        return employment


def fill_user_region(user):
    region = user['REGION']
    if pd.isnull(region):
        return 'NA'
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
    users_df['LIST_OWN_CLEANED'] = users_df.apply(clean_list_playback, axis=1, args=('LIST_OWN',))
    users_df['LIST_BACK_CLEANED'] = users_df.apply(clean_list_playback, axis=1, args=('LIST_BACK',))

    users_list_own_mean = users_df['LIST_OWN_CLEANED'].mean()
    users_list_back_mean = users_df['LIST_BACK_CLEANED'].mean()

    users_df['LIST_OWN_CLEANED'] = users_df.apply(fill_empty_list_playback, axis=1, args=('LIST_OWN_CLEANED', users_list_own_mean))
    users_df['LIST_BACK_CLEANED'] = users_df.apply(fill_empty_list_playback, axis=1, args=('LIST_BACK_CLEANED', users_list_back_mean))

    return users_df


def clean_questions(users_df):
    for i in range (1, 20):
        col_name = 'Q' + str(i)
        q_mean = users_df[col_name].mean()
        users_df[col_name + '_CLEANED'] = users_df.apply(fill_empty_question, axis=1, args=(col_name, q_mean))

    return users_df


def encode_user_music(user):
    music_importance = user['MUSIC']
    return users_encoding.MUSIC_ENCODING_MAP[music_importance]


def write_users_df_to_csv(users_df):
    USERS_CLEANED_DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'users_cleaned.csv')
    users_df.to_csv(USERS_CLEANED_DATA_FILE_PATH, sep=',', encoding='utf-8')


if __name__ == '__main__':
    print("Loading users..")
    users_df = load_users_data()

    print("Cleaning users age..")
    users_df['AGE_CLEANED'] = users_df.apply(fill_user_age, axis=1)

    print("Cleaning working..")
    users_df['WORKING_CLEANED'] = users_df.apply(fill_user_working, axis=1)

    print("Cleaning region..")
    users_df['REGION_CLEANED'] = users_df.apply(fill_user_region, axis=1)

    print("Encoding music..")
    users_df['MUSIC_ENCODED'] = users_df.apply(encode_user_music, axis=1)

    print("Cleaning list own and list back..")
    users_df = clean_list_own_and_back(users_df)

    print("Cleaning questions..")
    users_df = clean_questions(users_df)

    print("Writing cleaned data to CSV file..")
    write_users_df_to_csv(users_df)