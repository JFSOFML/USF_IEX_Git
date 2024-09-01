import pandas as pd
import sqlite3

# Load Titanic_data to dataframe
titanic_df = pd.read_csv("Data_files/titanic_Data.csv")

# Select columns to load for the Ames Housing dataset
columns = [
    "Overall Qual",
    "Overall Cond",
    "Gr Liv Area",
    "Central Air",
    "Total Bsmt SF",
    "SalePrice",
    "Lot Area",
    "Full Bath",
    "Half Bath",
    "TotRms AbvGrd",
    "Fireplaces",
    "Wood Deck SF",
]

# Load the Ames Housing dataset to dataframe
housing_df = pd.read_csv("Data_files/AmesHousing.txt", sep="\t", usecols=columns)

# Loading movie data into a dataframe.
movie_df = pd.read_csv("Data_files/movie_data.csv")

# Create or Connect to Database
conn = sqlite3.connect("DSRA_projects.db")

# Write DataFrame to Database
titanic_df.to_sql("titanic", conn, if_exists="replace", index=False)
housing_df.to_sql("housing", conn, if_exists="replace", index=False)
movie_df.to_sql("movie", conn, if_exists="replace", index=False)

# Close Connection
conn.close()
