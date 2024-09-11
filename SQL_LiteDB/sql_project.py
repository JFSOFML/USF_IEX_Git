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

    # opening the CSV
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)

        #create sql table if the it doesn't already exists
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table}\
              ({','.join([f'{header} TEXT' for header in headers])});"
        cursor.execute(create_table_sql)
        # Inert data into table
        insert_data_sql = f"INSERT INTO {table}\
              VALUES ({', '.join(['?' for _ in headers])})"
        # ? is a markdown wild character "List comprehension"
        for row in reader:
            cursor.execute(insert_data_sql, row)# iterates 1 row at a time, seperates and joins.

    conn.commit()
    conn.close()







titanic_db = "DB_files/titanic.db"
titanic_table = "titanic"
titanic_csv = "CSV_files/titanic_data.csv"


csv_to_sqlite(titanic_csv, titanic_db, titanic_table)



conn = sqlite3.connect(titanic_db)
query = f"SELECT * FROM {titanic_table} WHERE Survived = 0 LIMIT 10"

df = pd.read_sql_query(query, conn)

#End of File (EOF)
