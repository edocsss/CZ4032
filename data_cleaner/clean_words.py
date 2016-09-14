import pandas as pd
import os
import encoding.users as users_encoding


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


def fill_empty_own_artist_music(word):
    own_artist_music= word['OWN_ARTIST_MUSIC']
    if pd.isnull(own_artist_music):
        return 'Own none of their music'
    else:
        return own_artist_music


def clean_like_artist(words_df):
    like_artist_mean = words_df['LIKE_ARTIST'].mean()
    words_df['LIKE_ARTIST_CLEANED'] = words_df.apply(fill_empty_like_artist, axis=1, args=(like_artist_mean,))
    return words_df


def fill_empty_like_artist(word, like_artist_mean):
    own_artist_music = word['OWN_ARTIST_MUSIC']
    if own_artist_music in ['Own a little of their music', 'Own a lot of their music', 'Own all or most of their music']:
        return like_artist_mean
    else:
        return 0


def fill_empty_adjectives(word, adj):
    adjective = word[adj]
    if pd.isnull(adjective):
        return 2
    else:
        return int(adjective)


def write_users_df_to_csv(words_df):
    WORDS_CLEANED_DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'words_cleaned.csv')
    words_df.to_csv(WORDS_CLEANED_DATA_FILE_PATH, sep=',', encoding='utf-8')


if __name__ == '__main__':
    print("Loading words..")
    words_df = load_words_data()

    print("Filling in empty heard of..")
    words_df['HEARD_OF_CLEANED'] = words_df.apply(fill_empty_heard_of, axis=1)

    print("Filling in empty own artist music..")
    words_df['OWN_ARTIST_MUSIC_CLEANED'] = words_df.apply(fill_empty_heard_of, axis=1)

    print("Filling in empty like artist..")
    words_df = clean_like_artist(words_df)

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
        'Good Lyrics',
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
        words_df[adj + '_CLEANED'] = words_df.apply(fill_empty_adjectives, axis=1, args=(adj,))


    print("Writing cleaned data to CSV file..")
    write_users_df_to_csv(words_df)

    print()
    print(words_df.head(100))