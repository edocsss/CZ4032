import mysql.connector
import mysql.connector.conversion
from mysql.connector import errorcode
import json
import os


f = open(os.path.join(os.getcwd(), 'mysql_config.json'), 'r')
CONNECTION_INFO = json.load(f)
f.close()

CREATE_TRAIN_TABLE_SQL = """
    CREATE TABLE `train` (
        `artist` int(100),
        `track` int(100),
        `user` int(100),
        `rating` int(100),
        `time` int(100),
        PRIMARY KEY (artist, track, user)
    )
"""

CREATE_USERS_TABLE_SQL = """
    CREATE TABLE `users` (
        `id` int(100) PRIMARY KEY,
        `respid` int(100),
        `gender` varchar(50),
        `age` int(100),
        `working` varchar(100),
        `region` varchar(50),
        `music` varchar(100),
        `list_own` float,
        `list_back` float,
        `q1` int(100),
        `q2` int(100),
        `q3` int(100),
        `q4` int(100),
        `q5` int(100),
        `q6` int(100),
        `q7` int(100),
        `q8` int(100),
        `q9` int(100),
        `q10` int(100),
        `q11` int(100),
        `q12` int(100),
        `q13` int(100),
        `q14` int(100),
        `q15` int(100),
        `q16` int(100),
        `q17` int(100),
        `q18` int(100),
        `q19` int(100)
    )
"""

CREATE_WORDS_TABLE_SQL = """
    CREATE TABLE `words` (
        `id` int(50) PRIMARY KEY,
        `artist` int(50) NOT NULL,
        `user` int(50) NOT NULL,
        `heard_of` varchar(50),
        `own_artist_music` varchar(100),
        `like_artist` int(50),
        `uninspired` int(5),
        `sophisticated` int(5),
        `aggressive` int(5),
        `edgy` int(5),
        `sociable` int(5),
        `laid_back` int(5),
        `wholesome` int(5),
        `uplifting` int(5),
        `intriguing` int(5),
        `legendary` int(5),
        `free` int(5),
        `thoughtful` int(5),
        `outspoken` int(5),
        `serious` int(5),
        `good_lyrics` int(5),
        `unattractive` int(5),
        `confident` int(5),
        `old` int(5),
        `youthful` int(5),
        `boring` int(5),
        `current` int(5),
        `colourful` int(5),
        `stylish` int(5),
        `cheap` int(5),
        `irrelevant` int(5),
        `heartfelt` int(5),
        `calm` int(5),
        `pioneer` int(5),
        `outgoing` int(5),
        `inspiring` int(5),
        `beautiful` int(5),
        `fun` int(5),
        `authentic` int(5),
        `credible` int(5),
        `way_out` int(5),
        `cool` int(5),
        `catchy` int(5),
        `sensitive` int(5),
        `mainstream` int(5),
        `superficial` int(5),
        `annoying` int(5),
        `dark` int(5),
        `passionate` int(5),
        `not_authentic` int(5),
        `background` int(5),
        `timeless` int(5),
        `depressing` int(5),
        `original` int(5),
        `talented` int(5),
        `worldly` int(5),
        `distinctive` int(5),
        `approachable` int(5),
        `genius` int(5),
        `trendsetter` int(5),
        `noisy` int(5),
        `upbeat` int(5),
        `relatable` int(5),
        `energetic` int(5),
        `exciting` int(5),
        `emotional` int(5),
        `nostalgic` int(5),
        `none_of_these` int(5),
        `progressive` int(5),
        `sexy` int(5),
        `over` int(5),
        `rebellious` int(5),
        `fake` int(5),
        `cheesy` int(5),
        `popular` int(5),
        `superstar` int(5),
        `relaxed` int(5),
        `intrusive` int(5),
        `unoriginal` int(5),
        `dated` int(5),
        `iconic` int(5),
        `unapproachable` int(5),
        `classic` int(5),
        `playful` int(5),
        `arrogant` int(5),
        `warm` int(5),
        `soulful` int(5)
    )
"""

INSERT_TRAIN_SQL = """
    INSERT INTO train (
        artist,
        track,
        user,
        rating,
        time
    ) VALUES (
        %(artist)s,
        %(track)s,
        %(user)s,
        %(rating)s,
        %(time)s
    )
"""

INSERT_USER_SQL = """
    INSERT INTO users (
        id,
        respid,
        gender,
        age,
        working,
        region,
        music,
        list_own,
        list_back,
        q1,
        q2,
        q3,
        q4,
        q5,
        q6,
        q7,
        q8,
        q9,
        q10,
        q11,
        q12,
        q13,
        q14,
        q15,
        q16,
        q17,
        q18,
        q19
    ) VALUES (
        %(id)s,
        %(respid)s,
        %(gender)s,
        %(age)s,
        %(working)s,
        %(region)s,
        %(music)s,
        %(list_own)s,
        %(list_back)s,
        %(q1)s,
        %(q2)s,
        %(q3)s,
        %(q4)s,
        %(q5)s,
        %(q6)s,
        %(q7)s,
        %(q8)s,
        %(q9)s,
        %(q10)s,
        %(q11)s,
        %(q12)s,
        %(q13)s,
        %(q14)s,
        %(q15)s,
        %(q16)s,
        %(q17)s,
        %(q18)s,
        %(q19)s
    )
"""


INSERT_WORD_SQL = """
    INSERT INTO words (
        `id`,
        `artist`,
        `user`,
        `heard_of`,
        `own_artist_music`,
        `like_artist`,
        `uninspired`,
        `sophisticated`,
        `aggressive`,
        `edgy`,
        `sociable`,
        `laid_back`,
        `wholesome`,
        `uplifting`,
        `intriguing`,
        `legendary`,
        `free`,
        `thoughtful`,
        `outspoken`,
        `serious`,
        `good_lyrics`,
        `unattractive`,
        `confident`,
        `old`,
        `youthful`,
        `boring`,
        `current`,
        `colourful`,
        `stylish`,
        `cheap`,
        `irrelevant`,
        `heartfelt`,
        `calm`,
        `pioneer`,
        `outgoing`,
        `inspiring`,
        `beautiful`,
        `fun`,
        `authentic`,
        `credible`,
        `way_out`,
        `cool`,
        `catchy`,
        `sensitive`,
        `mainstream`,
        `superficial`,
        `annoying`,
        `dark`,
        `passionate`,
        `not_authentic`,
        `background`,
        `timeless`,
        `depressing`,
        `original`,
        `talented`,
        `worldly`,
        `distinctive`,
        `approachable`,
        `genius`,
        `trendsetter`,
        `noisy`,
        `upbeat`,
        `relatable`,
        `energetic`,
        `exciting`,
        `emotional`,
        `nostalgic`,
        `none_of_these`,
        `progressive`,
        `sexy`,
        `over`,
        `rebellious`,
        `fake`,
        `cheesy`,
        `popular`,
        `superstar`,
        `relaxed`,
        `intrusive`,
        `unoriginal`,
        `dated`,
        `iconic`,
        `unapproachable`,
        `classic`,
        `playful`,
        `arrogant`,
        `warm`,
        `soulful`
    ) VALUES (
        %(id)s,
        %(artist)s,
        %(user)s,
        %(heard_of)s,
        %(own_artist_music)s,
        %(like_artist)s,
        %(uninspired)s,
        %(sophisticated)s,
        %(aggressive)s,
        %(edgy)s,
        %(sociable)s,
        %(laid_back)s,
        %(wholesome)s,
        %(uplifting)s,
        %(intriguing)s,
        %(legendary)s,
        %(free)s,
        %(thoughtful)s,
        %(outspoken)s,
        %(serious)s,
        %(good_lyrics)s,
        %(unattractive)s,
        %(confident)s,
        %(old)s,
        %(youthful)s,
        %(boring)s,
        %(current)s,
        %(colourful)s,
        %(stylish)s,
        %(cheap)s,
        %(irrelevant)s,
        %(heartfelt)s,
        %(calm)s,
        %(pioneer)s,
        %(outgoing)s,
        %(inspiring)s,
        %(beautiful)s,
        %(fun)s,
        %(authentic)s,
        %(credible)s,
        %(way_out)s,
        %(cool)s,
        %(catchy)s,
        %(sensitive)s,
        %(mainstream)s,
        %(superficial)s,
        %(annoying)s,
        %(dark)s,
        %(passionate)s,
        %(not_authentic)s,
        %(background)s,
        %(timeless)s,
        %(depressing)s,
        %(original)s,
        %(talented)s,
        %(worldly)s,
        %(distinctive)s,
        %(approachable)s,
        %(genius)s,
        %(trendsetter)s,
        %(noisy)s,
        %(upbeat)s,
        %(relatable)s,
        %(energetic)s,
        %(exciting)s,
        %(emotional)s,
        %(nostalgic)s,
        %(none_of_these)s,
        %(progressive)s,
        %(sexy)s,
        %(over)s,
        %(rebellious)s,
        %(fake)s,
        %(cheesy)s,
        %(popular)s,
        %(superstar)s,
        %(relaxed)s,
        %(intrusive)s,
        %(unoriginal)s,
        %(dated)s,
        %(iconic)s,
        %(unapproachable)s,
        %(classic)s,
        %(playful)s,
        %(arrogant)s,
        %(warm)s,
        %(soulful)s
    )
"""


class MySQLConverter(mysql.connector.conversion.MySQLConverter):
    def _float32_to_mysql(self, value):
        return float(value)

    def _float64_to_mysql(self, value):
        return float(value)

    def _int32_to_mysql(self, value):
        return int(value)

    def _int64_to_mysql(self, value):
        return int(value)


def get_db_connection():
    conn = mysql.connector.connect(
        user=CONNECTION_INFO['user'],
        password=CONNECTION_INFO['password'],
        host=CONNECTION_INFO['host'],
        port=CONNECTION_INFO['port'],
        database=CONNECTION_INFO['db']
    )

    conn.set_converter_class(MySQLConverter)
    return conn


def close_db_connection(conn):
    conn.close()


def get_cursor(conn):
    return conn.cursor()


def close_cursor(cursor):
    return cursor.close()


def is_db_exist(conn):
    try:
        conn.database = CONNECTION_INFO['db']
        return True

    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_BAD_DB_ERROR:
            return False

    return False


def create_train_table(cursor):
    try:
        print("Creating table: train")
        cursor.execute(CREATE_TRAIN_TABLE_SQL)

    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("Table train already exists!")
            print()

        else:
            print(e.msg)
            print()


def create_users_table(cursor):
    try:
        print("Creating table: users")
        cursor.execute(CREATE_USERS_TABLE_SQL)

    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("Table users already exists!")
            print()

        else:
            print(e.msg)
            print()


def create_words_table(cursor):
    try:
        print("Creating table: words")
        cursor.execute(CREATE_WORDS_TABLE_SQL)

    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("Table words already exists!")
            print()

        else:
            print(e.msg)
            print()


def insert_train(cursor, train):
    try:
        cursor.execute(INSERT_TRAIN_SQL, train)

    except mysql.connector.Error as e:
        print(e.msg)
        print()


def insert_user(cursor, user):
    try:
        cursor.execute(INSERT_USER_SQL, user)

    except mysql.connector.Error as e:
        print(e.msg)
        print()


def insert_word(cursor, word):
    try:
        cursor.execute(INSERT_WORD_SQL, word)

    except mysql.connector.Error as e:
        print(e.msg)
        print()