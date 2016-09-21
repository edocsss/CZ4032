import os

import pandas as pd

DATA_DIR_PATH = '../data'


def load_train_data():
    TRAIN_DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'train.csv')
    train_df = pd.read_csv(TRAIN_DATA_FILE_PATH)
    return train_df


def load_users_data():
    USERS_DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'users_cleaned_2.csv')
    users_df = pd.read_csv(USERS_DATA_FILE_PATH)
    return users_df


def load_words_data():
    MUSICS_DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'words_cleaned.csv')
    musics_df = pd.read_csv(MUSICS_DATA_FILE_PATH)
    return musics_df


def convert_train_series_to_dict(i, train_series):
    return {
        'artist': train_series['Artist'],
        'track': train_series['Track'],
        'user': train_series['User'],
        'rating': train_series['Rating'],
        'time': train_series['Time']
    }


def convert_users_series_to_dict(i, user_series):
    return {
        'id': i,
        'respid': user_series['RESPID'],
        'gender': user_series['GENDER'],
        'age': user_series['AGE'],
        'working': user_series['WORKING'],
        'region': user_series['REGION'],
        'music': user_series['MUSIC'],
        'list_own': user_series['LIST_OWN'],
        'list_back': user_series['LIST_BACK'],
        'q1': user_series['Q1'],
        'q2': user_series['Q2'],
        'q3': user_series['Q3'],
        'q4': user_series['Q4'],
        'q5': user_series['Q5'],
        'q6': user_series['Q6'],
        'q7': user_series['Q7'],
        'q8': user_series['Q8'],
        'q9': user_series['Q9'],
        'q10': user_series['Q10'],
        'q11': user_series['Q11'],
        'q12': user_series['Q12'],
        'q13': user_series['Q13'],
        'q14': user_series['Q14'],
        'q15': user_series['Q15'],
        'q16': user_series['Q16'],
        'q17': user_series['Q17'],
        'q18': user_series['Q18'],
        'q19': user_series['Q19']
    }


def convert_words_series_to_dict(i, word_series):
    return {
        'id': i,
        'artist': word_series['Artist'],
        'user': word_series['User'],
        'heard_of': word_series['HEARD_OF'],
        'own_artist_music': word_series['OWN_ARTIST_MUSIC'],
        'like_artist': word_series['LIKE_ARTIST'],
        'uninspired': word_series['Uninspired'],
        'sophisticated': word_series['Sophisticated'],
        'aggressive': word_series['Aggressive'],
        'edgy': word_series['Edgy'],
        'sociable': word_series['Sociable'],
        'laid_back': word_series['Laid back'],
        'wholesome': word_series['Wholesome'],
        'uplifting': word_series['Uplifting'],
        'intriguing': word_series['Intriguing'],
        'legendary': word_series['Legendary'],
        'free': word_series['Free'],
        'thoughtful': word_series['Thoughtful'],
        'outspoken': word_series['Outspoken'],
        'serious': word_series['Serious'],
        'good_lyrics': word_series['Good lyrics'],
        'unattractive': word_series['Unattractive'],
        'confident': word_series['Confident'],
        'old': word_series['Old'],
        'youthful': word_series['Youthful'],
        'boring': word_series['Boring'],
        'current': word_series['Current'],
        'colourful': word_series['Colourful'],
        'stylish': word_series['Stylish'],
        'cheap': word_series['Cheap'],
        'irrelevant': word_series['Irrelevant'],
        'heartfelt': word_series['Heartfelt'],
        'calm': word_series['Calm'],
        'pioneer': word_series['Pioneer'],
        'outgoing': word_series['Outgoing'],
        'inspiring': word_series['Inspiring'],
        'beautiful': word_series['Beautiful'],
        'fun': word_series['Fun'],
        'authentic': word_series['Authentic'],
        'credible': word_series['Credible'],
        'way_out': word_series['Way out'],
        'cool': word_series['Cool'],
        'catchy': word_series['Catchy'],
        'sensitive': word_series['Sensitive'],
        'mainstream': word_series['Mainstream'],
        'superficial': word_series['Superficial'],
        'annoying': word_series['Annoying'],
        'dark': word_series['Dark'],
        'passionate': word_series['Passionate'],
        'not_authentic': word_series['Not authentic'],
        'background': word_series['Background'],
        'timeless': word_series['Timeless'],
        'depressing': word_series['Depressing'],
        'original': word_series['Original'],
        'talented': word_series['Talented'],
        'worldly': word_series['Worldly'],
        'distinctive': word_series['Distinctive'],
        'approachable': word_series['Approachable'],
        'genius': word_series['Genius'],
        'trendsetter': word_series['Trendsetter'],
        'noisy': word_series['Noisy'],
        'upbeat': word_series['Upbeat'],
        'relatable': word_series['Relatable'],
        'energetic': word_series['Energetic'],
        'exciting': word_series['Exciting'],
        'emotional': word_series['Emotional'],
        'nostalgic': word_series['Nostalgic'],
        'none_of_these': word_series['None of these'],
        'progressive': word_series['Progressive'],
        'sexy': word_series['Sexy'],
        'over': word_series['Over'],
        'rebellious': word_series['Rebellious'],
        'fake': word_series['Fake'],
        'cheesy': word_series['Cheesy'],
        'popular': word_series['Popular'],
        'superstar': word_series['Superstar'],
        'relaxed': word_series['Relaxed'],
        'intrusive': word_series['Intrusive'],
        'unoriginal': word_series['Unoriginal'],
        'dated': word_series['Dated'],
        'iconic': word_series['Iconic'],
        'unapproachable': word_series['Unapproachable'],
        'classic': word_series['Classic'],
        'playful': word_series['Playful'],
        'arrogant': word_series['Arrogant'],
        'warm': word_series['Warm'],
        'soulful': word_series['Soulful']
    }


