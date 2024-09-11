"""SQL DB project"""
import csv
import sqlite3
import os
import pandas as pd


def csv_to_sqlite(csv_file, db_file, table):
    """Imports data from a CSV file into an SQLite DB table"""
    if os.path.exists(db_file):
        print("already exists")
        return

    conn = sqlite3.connect(db_file)

    cursor = conn.cursor()

    cursor.execute(f"DROP TABLE IF EXISTS {table}")

    # opening the CSV with encoding
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)

        # Create SQL table if it doesn't already exist
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table}\
              ({','.join([f'{header} TEXT' for header in headers])});"
        cursor.execute(create_table_sql)

        # Insert data into table
        insert_data_sql = f"INSERT INTO {table} VALUES\
              ({', '.join(['?' for _ in headers])})"
        for row in reader:
            cursor.execute(insert_data_sql, row)

    conn.commit()
    conn.close()


# Updated constant names to UPPER_CASE
TITANIC_DB = "DB_files/titanic.db"
TITANIC_TABLE = "titanic"
TITANIC_CSV = "CSV_files/titanic_data.csv"

# Call the function to load data from CSV to SQLite
csv_to_sqlite(TITANIC_CSV, TITANIC_DB, TITANIC_TABLE)

# Connect to the SQLite database and perform a query
db_conn = sqlite3.connect(TITANIC_DB)
query = f"SELECT * FROM {TITANIC_TABLE} WHERE Survived = 0 LIMIT 10"

df = pd.read_sql_query(query, db_conn)
db_conn.close()  # Close the connection after the query

# End of File (EOF)
