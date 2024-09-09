"""
This Flask application serves predictions and runs SQL queries
using pretrained models and a SQLite database.
"""

import json  # Standard library imports should come first
import sqlite3
import pickle
from flask import Flask, request, jsonify
import pandas as pd
import werkzeug.exceptions

# Initialize the Flask application
app = Flask(__name__)

# Load the model from the pickle file
with open("models/Scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/forest.pkl", "rb") as f:
    forest = pickle.load(f)
with open("models/SVCModel_pipeline.pkl", "rb") as f:
    SVCModel_pipeline = pickle.load(f)

def execute_query(sql_query):
    """
    Execute an SQL query and return the results.

    Args:
        sql_query (str): The SQL query to execute.

    Returns:
        dict: A dictionary with the query results or error message.
    """
    try:
        conn = sqlite3.connect("/app/DSRA_projects.db")
        cursor = conn.cursor()
        cursor.execute(sql_query)
        columns = [description[0] for description in cursor.description]
        data = cursor.fetchall()
        conn.close()
        return {"columns": columns, "data": data}
    except sqlite3.DatabaseError as e:
        return {"error": str(e)}

@app.route("/predict_titanic", methods=["POST"])
def predict_titanic():
    """
    This function takes in an array of data and uses it to
    predict against a pretrained model for the Titanic data.
    """
    data = request.json
    df = pd.DataFrame([data])
    prediction = SVCModel_pipeline.predict(df)[0]
    return jsonify({"Survived": int(prediction)})

@app.route("/predict_housing", methods=["POST"])
def predict_housing():
    """
    This function takes in an array of data and uses it to
    predict against a pretrained model for the Housing data.
    """
    data = request.json
    df = pd.DataFrame([data])
    scaled_data = scaler.transform(df)
    prediction = forest.predict(scaled_data)[0]
    return jsonify({"price": float(prediction)})

@app.route("/query", methods=["POST"])
def query():
    """
    Run an SQL query provided in the request body.

    Returns:
        JSON response with the query result or error message.
    """
    try:
        request_data = request.json
        sql_query = request_data.get("query")
        if not sql_query:
            return jsonify({"error": "No query provided"}), 400
        result = execute_query(sql_query)
        return jsonify(result)
    except sqlite3.DatabaseError as e:  # Specific database exception
        app.logger.error("SQL Error: %s", e)
        return jsonify({"error": str(e)}), 500
    except werkzeug.exceptions.BadRequest as e:  # Specific Flask-related error
        app.logger.error("Request Error: %s", e)
        return jsonify({"error": str(e)}), 400
    except json.JSONDecodeError as e:  # Handle JSON decoding error
        app.logger.error("JSON Decode Error: %s", e)
        return jsonify({"error": "Invalid JSON format"}), 400
