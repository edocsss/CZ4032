import pandas as pd
import os
import encoding

DATA_DIR_PATH = '../data'


def load_words_data():
    MUSICS_DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'words.csv')
    musics_df = pd.read_csv(MUSICS_DATA_FILE_PATH)
    return musics_df


def fill_empty_heard_of(word):
    heard_of = word['HEARD_OF']
    if pd.isnull(heard_of):
        return 'Never heard of'
    else:
        return heard_of


def encode_word(word, use_binary=False):
    heard_of = word['HEARD_OF'].lower()
    own_artist_music = word['OWN_ARTIST_MUSIC'].lower()

    if use_binary:
        heard_of_encoded = encoding.HEARD_OF_BINARY_MAP[heard_of]
        own_artist_music_encoded = encoding.OWN_ARTIST_BINARY_MAP[own_artist_music]
    else:
        heard_of_encoded = encoding.HEARD_OF_ONEHOT_MAP[heard_of]
        own_artist_music_encoded = encoding.OWN_ARTIST_ONEHOT_MAP[own_artist_music]

    for i in range(0, len(heard_of_encoded)):
        word['HEARD_OF_' + str(i)] = heard_of_encoded[i]

    for j in range(0, len(own_artist_music_encoded)):
        word['OWN_ARTIST_MUSIC_' + str(j)] = own_artist_music_encoded[j]

    return word


def fill_empty_own_artist_music(word):
    own_artist_music= word['OWN_ARTIST_MUSIC']
    if pd.isnull(own_artist_music):
        return 'Own none of their music'
    else:
        return own_artist_music


def clean_like_artist(words_df):
    like_artist_mean = words_df['LIKE_ARTIST'].mean()
    words_df['LIKE_ARTIST'] = words_df.apply(fill_empty_like_artist, axis=1, args=(like_artist_mean,))
    return words_df


def fill_empty_like_artist(word, like_artist_mean):
    like_artist = word['LIKE_ARTIST']
    if not pd.isnull(like_artist):
        return like_artist
    else:
        return like_artist_mean


def fill_empty_adjectives(word, adj):
    adjective = word[adj]
    if pd.isnull(adjective):
        return 2
    else:
        return adjective


def write_users_df_to_csv(words_df):
    WORDS_CLEANED_DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'words_cleaned_onehot.csv')
    words_df.to_csv(WORDS_CLEANED_DATA_FILE_PATH, sep=',', encoding='utf-8')


if __name__ == '__main__':
    print("Loading words..")
    words_df = load_words_data()

    print("Dropping 'Good Lyrics' column since it is duplicated..")
    words_df = words_df.drop('Good Lyrics', axis=1)

    print("Filling in empty heard of..")
    words_df['HEARD_OF'] = words_df.apply(fill_empty_heard_of, axis=1)

    print("Filling in empty own artist music..")
    words_df['OWN_ARTIST_MUSIC'] = words_df.apply(fill_empty_own_artist_music, axis=1)

    print("Filling in empty like artist..")
    words_df = clean_like_artist(words_df)

    print("Encoding non integer fields..")
    words_df = words_df.apply(encode_word, axis=1, args=(False,))

    adjectives = [
        'Uninspired',
        'Sophisticated',
        'Aggressive',
        'Edgy',
        'Sociable',
        'Laid back',
        'Wholesome',
        'Uplifting',
        'Intriguing',
        'Legendary',
        'Free',
        'Thoughtful',
        'Outspoken',
        'Serious',
        'Good lyrics',
        'Unattractive',
        'Confident',
        'Old',
        'Youthful',
        'Boring',
        'Current',
        'Colourful',
        'Stylish',
        'Cheap',
        'Irrelevant',
        'Heartfelt',
        'Calm',
        'Pioneer',
        'Outgoing',
        'Inspiring',
        'Beautiful',
        'Fun',
        'Authentic',
        'Credible',
        'Way out',
        'Cool',
        'Catchy',
        'Sensitive',
        'Mainstream',
        'Superficial',
        'Annoying',
        'Dark',
        'Passionate',
        'Not authentic',
        'Background',
        'Timeless',
        'Depressing',
        'Original',
        'Talented',
        'Worldly',
        'Distinctive',
        'Approachable',
        'Genius',
        'Trendsetter',
        'Noisy',
        'Upbeat',
        'Relatable',
        'Energetic',
        'Exciting',
        'Emotional',
        'Nostalgic',
        'None of these',
        'Progressive',
        'Sexy',
        'Over',
        'Rebellious',
        'Fake',
        'Cheesy',
        'Popular',
        'Superstar',
        'Relaxed',
        'Intrusive',
        'Unoriginal',
        'Dated',
        'Iconic',
        'Unapproachable',
        'Classic',
        'Playful',
        'Arrogant',
        'Warm',
        'Soulful'
    ]

    for adj in adjectives:
        print("Filling in empty '{}'..".format(adj))
        words_df[adj] = words_df.apply(fill_empty_adjectives, axis=1, args=(adj,))

    print("Writing cleaned data to CSV file..")
    write_users_df_to_csv(words_df)
