import data_parser.db_helper as db_helper
import data_parser.csv_parser as csv_parser
from datetime import date, datetime

if __name__ == '__main__':
    conn = db_helper.get_db_connection()
    cursor = db_helper.get_cursor(conn)

    db_helper.create_database(conn, cursor)
    db_helper.create_post_table(cursor)

    # Read XML and store to DB
    # xml_parser.parse_post_xml_and_store_db(conn, cursor, n=100)

    db_helper.close_cursor(cursor)
    db_helper.close_db_connection(conn)