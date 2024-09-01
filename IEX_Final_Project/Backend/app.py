import sqlite3
import pickle
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the model from the pickle file
with open("models/Scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/forest.pkl", "rb") as f:
    forest = pickle.load(f)    
with open("models/SVCModel_pipeline.pkl", "rb") as f:
    SVCModel_pipeline = pickle.load(f)    

# Function to run a SQL query
def execute_query(query):
    try:
        # Connect to the SQLite database file
        conn = sqlite3.connect("/app/DSRA_projects.db")

        # Create a cursor to execute the query
        cursor = conn.cursor()

        # Execute the given SQL query
        cursor.execute(query)

        # Get the column names from the result
        columns = [description[0] for description in cursor.description]

        # Fetch all rows of data from the query result
        data = cursor.fetchall()

        # Close the database connection
        conn.close()

        # Return the column names and data
        return {"columns": columns, "data": data}
    except Exception as e:
        # If there's an error, return the error message
        return {"error": str(e)}

@app.route("/predict_titanic", methods=["POST"])
def predict_titanic():
    """
    This function takes in an array of data and uses it to
    predict against a pretrained model for the titanic data.
    """
    data = request.json
    # df = pd.DataFrame(data, index=[0])
    df = pd.DataFrame([data])
    prediction = SVCModel_pipeline.predict(df)[0]
    return jsonify({"price": float(prediction)})


@app.route("/predict_housing", methods=["POST"])
def predict_housing():
    """
    This function takes in an array of data and uses it to
    predict against a pretrained model for the Housing data.
    """
    data = request.json
    # df = pd.DataFrame(data, index=[0])
    df = pd.DataFrame([data])
    scaled_data = scaler.transform(df)
    prediction = forest.predict(scaled_data)[0]
    survival_prob = forest.predict_proba(scaled_data)[0][1]
    return jsonify({"survived": int(prediction), "survival_prob": float(survival_prob)})
   

# Route to handle requests to /query
@app.route("/query", methods=["POST"])
def query():
    try:
        # Get the data sent in the request as JSON
        request_data = request.json

        # Extract the SQL query from the JSON data
        query = request_data.get("query")
        if not query:
            # If no query is provided, return an error
            return jsonify({"error": "No query provided"}), 400

        # Run the query using the execute_query function
        result = execute_query(query)

        # Return the query result as a JSON response
        return jsonify(result)
    except Exception as e:
        # If there's an error, log it and return the error message
        app.logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


# Start the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
