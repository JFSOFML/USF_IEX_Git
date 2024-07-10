## Importing CSV Data into SQLite Database

This script helps you import data from a CSV (Comma Separated Values) file into an SQLite database. It can be used with various datasets stored in CSV format.

**Step-by-Step Breakdown:**

1. **Import Libraries:**
`import csv`
`import sqlite3`
`import pandas as pd`
`import os`


   The script uses several libraries to work with CSV files, SQLite databases, and data manipulation. You can install them using `pip install csv pandas os` in your terminal or command prompt.(SQLite3 comes with python)

   - `csv`: This library helps you read and write data in CSV format.
   - `sqlite3`: This library allows you to interact with SQLite databases.
   - `pandas`: This library is powerful for data analysis and manipulation, especially useful after importing data.
   - `os`: This library provides functions for interacting with the operating system, like checking if a file exists.

2. **`csv_to_sqlite` Function:**

   The core functionality lies in the `csv_to_sqlite` function. It takes three arguments:

   - `csv_file`: Path to the CSV file you want to import.
   - `db_file`: Path to the SQLite database file you want to create or use.
   - `table`: Name of the table you want to create inside the database to store the CSV data.

   Here's a breakdown of what the function does:

     - Checks if the database file already exists. If it does, it prints a message.
     - Connects to the database file using `sqlite3.connect`.
     - Creates a cursor object to execute SQL commands.
     - Drops the table with the provided name (if it exists) to ensure a clean slate.
     - Opens the CSV file for reading and creates a reader object.
     - Reads the first row (usually containing headers) and stores them in a list.
     - Creates an SQL statement to create a table in the database with columns matching the CSV headers and data type set to `TEXT` (assuming mostly text data).
     - Executes the CREATE TABLE statement using the cursor.
     - Creates another SQL statement to insert data into the table, using question marks (`?`) as placeholders for each data point in a row.
     - Iterates through each row of the CSV data.
     - For each row, executes the INSERT statement with the current row data replacing the question marks.
     - Finally, commits the changes to the database and closes the connection.

3. **Importing Titanic Data (Example):**

   - The script defines variables for the database file path (`titanic_db`), table name (`titanic_table`), and CSV file path (`titanic_csv`).
   - It calls the `csv_to_sqlite` function with these defined variables to demonstrate importing sample Titanic data from the CSV file into the SQLite database.

4. **Querying Data (Optional):**

   - The script includes commented-out sections for connecting to the database and querying data. You can uncomment these lines to explore the imported data.
   - It defines a query to select the first 10 rows where the `Survived` column value is 0 (people who didn't survive).
   - It uses `pandas.read_sql_query` to execute the query and store the results in a pandas DataFrame (`df`). This DataFrame allows you to further analyze the data.

**How to Use:**

1. Install the required libraries (`csv`, `sqlite3`, `pandas`, and `os`) using `pip install`.
2. Update the script with your specific file paths for the CSV data and the desired database file.
3. Run the script using a Python interpreter.

**Additional Notes:**

- This script assumes the CSV file has headers in the first row.
- The script currently sets all columns to the `TEXT` data type in the database. You might need to modify this based on your specific data types.
- After the data is imported, you can use pandas (the `df` variable) to explore and analyze the data in the DataFrame.
