import data_parser.db_helper as db_helper
import data_parser.csv_parser as csv_parser
from datetime import date, datetime

if __name__ == '__main__':
    conn = db_helper.get_db_connection()
    cursor = db_helper.get_cursor(conn)

    db_helper.create_users_table(cursor)
    db_helper.create_words_table(cursor)

    # Read users data and store to DB
    users_df = csv_parser.load_users_data()
    for i, user_series in users_df.iterrows():
        user = csv_parser.convert_users_series_to_dict(i, user_series)
        db_helper.insert_user(cursor, user)

    conn.commit()

    # Read words data and store to DB
    words_df = csv_parser.load_words_data()
    for i, word_series in words_df.iterrows():
        word = csv_parser.convert_words_series_to_dict(i, word_series)
        db_helper.insert_word(cursor, word)

    conn.commit()

    # db_helper.close_cursor(cursor)
    db_helper.close_db_connection(conn)