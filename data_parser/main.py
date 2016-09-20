import data_parser.db_helper as db_helper
import data_parser.csv_parser as csv_parser
from datetime import date, datetime


def parse_and_store_train_to_db(conn, cursor):
    train_df = csv_parser.load_train_data()
    print("Train DataFrame size: {}".format(train_df.shape[0]))

    for i, train_series in train_df.iterrows():
        print(i)
        train = csv_parser.convert_train_series_to_dict(i, train_series)
        db_helper.insert_train(cursor, train)

    conn.commit()
    print("DONE WITH TRAIN!!")
    print()
    print()


def parse_and_store_users_to_db(conn, cursor):
    users_df = csv_parser.load_users_data()
    print("Users DataFrame size: {}".format(users_df.shape[0]))

    for i, user_series in users_df.iterrows():
        print(i)
        user = csv_parser.convert_users_series_to_dict(i, user_series)
        db_helper.insert_user(cursor, user)

    conn.commit()
    print("DONE WITH USERS!!")
    print()
    print()


def parse_and_store_words_to_db(conn, cursor):
    words_df = csv_parser.load_words_data()
    print("Words DataFrame size: {}".format(words_df.shape[0]))

    for i, word_series in words_df.iterrows():
        print(i)
        word = csv_parser.convert_words_series_to_dict(i, word_series)
        db_helper.insert_word(cursor, word)

    conn.commit()
    print("DONE WITH WORDS!!")


if __name__ == '__main__':
    conn = db_helper.get_db_connection()
    cursor = db_helper.get_cursor(conn)

    db_helper.create_train_table(cursor)
    db_helper.create_users_table(cursor)
    db_helper.create_words_table(cursor)

    parse_and_store_train_to_db(conn, cursor)
    parse_and_store_users_to_db(conn, cursor)
    parse_and_store_words_to_db(conn, cursor)

    db_helper.close_cursor(cursor)
    db_helper.close_db_connection(conn)