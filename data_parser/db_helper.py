import mysql.connector
from mysql.connector import errorcode


DB_NAME = 'CZ4045_NLP'
CONNECTION_INFO = {
    'user': 'admini9eesFs',
    'password': 'ue1tA9tcZ4JZ',
    'host': '127.0.0.1'
}

CREATE_DATABASE_SQL = "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(DB_NAME)

CREATE_POSTS_TABLE_SQL = """
    CREATE TABLE `posts` (
        `id` int(50) PRIMARY KEY,
        `post_type_id` int(50),
        `parent_id` int(50),
        `accepted_answer_id` int(50),
        `creation_date` date DEFAULT NULL,
        `score` int(50),
        `view_count` int(50),
        `body` varchar(5000),
        `owner_user_id` int(50),
        `last_editor_user_id` int(50),
        `last_editor_display_name` varchar(100),
        `last_edit_date` date DEFAULT NULL,
        `last_activity_date` date DEFAULT NULL,
        `title` varchar(500),
        `tags` varchar(500),
        `answer_count` int(50),
        `comment_count` int(50),
        `favorite_count` int(50)
    )
"""

INSERT_POST_SQL = """
    INSERT INTO posts (
        id,
        post_type_id,
        parent_id,
        accepted_answer_id,
        creation_date,
        score,
        view_count,
        body,
        owner_user_id,
        last_editor_user_id,
        last_editor_display_name,
        last_edit_date,
        last_activity_date,
        title,
        tags,
        answer_count,
        comment_count,
        favorite_count
    ) VALUES (
        %(id)s,
        %(post_type_id)s,
        %(parent_id)s,
        %(accepted_answer_id)s,
        %(creation_date)s,
        %(score)s,
        %(view_count)s,
        %(body)s,
        %(owner_user_id)s,
        %(last_editor_user_id)s,
        %(last_editor_display_name)s,
        %(last_edit_date)s,
        %(last_activity_date)s,
        %(title)s,
        %(tags)s,
        %(answer_count)s,
        %(comment_count)s,
        %(favorite_count)s
    )
"""


def get_db_connection():
    return mysql.connector.connect(user=CONNECTION_INFO['user'], host=CONNECTION_INFO['host'])


def close_db_connection(conn):
    conn.close()


def get_cursor(conn):
    return conn.cursor()


def close_cursor(cursor):
    return cursor.close()


def is_db_exist(conn):
    try:
        conn.database = DB_NAME
        return True

    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_BAD_DB_ERROR:
            return False

    return False


def create_database(conn, cursor):
    if not is_db_exist(conn):
        try:
            cursor.execute(CREATE_DATABASE_SQL)
            conn.database = DB_NAME

        except mysql.connector.Error as e:
            print("Failed creating database: {}".format(e))
            exit(1)


def create_post_table(cursor):
    try:
        print("Creating table: post")
        cursor.execute(CREATE_POSTS_TABLE_SQL)

    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("Table post already exists!")
            print()
        else:
            print(e.msg)
            print()

def insert_post(conn, cursor, post):
    try:
        cursor.execute(INSERT_POST_SQL, post)
        conn.commit()
        print("A new post just inserted!")
        print(post)
        print()

    except mysql.connector.Error as e:
        print(e.msg)
        print()