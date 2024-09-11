"""Flask Logic and route handling."""

import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)


def execute_query(query_input):
    """This function connects and executes the db query."""
    try:
        conn = sqlite3.connect("/app/titanic.db")
        cursor = conn.cursor()
        cursor.execute(query_input)
        columns = [description[0] for description in cursor.description]
        data = cursor.fetchall()
        conn.close()
        return {"columns": columns, "data": data}
    except sqlite3.DatabaseError as db_error:
        return jsonify({"error": f"Database error: {str(db_error)}"}), 500
    except ValueError as val_error:
        return jsonify({"error": f"Value error: {str(val_error)}"}), 400
    except KeyError as e:
        return jsonify({"error": f"KeyError: {str(e)}"}), 400  # Bad Request
    except IOError as e:
        return jsonify({"error": f"IOError: {str(e)}"}), 500  # Internal Server Error

@app.route("/query", methods=["POST"])
def query():

    """Route handling for any queries."""
    try:
        # GET: the JSON data from the POST request
        request_data = request.json

        # EXTRACT: the query from the JSON data
        query_data = request_data.get("query")
        if not query_data:
            return jsonify({"error": "No query provided"}), 400

        # EXECUTE: the query
        result = execute_query(query_data)

        # RETURN: the result as a JSON response
        return jsonify(result)
    except sqlite3.DatabaseError as db_error:
        return jsonify({"error": f"Database error: {str(db_error)}"}), 500
    except ValueError as val_error:
        return jsonify({"error": f"Value error: {str(val_error)}"}), 400
    except KeyError as e:
        return jsonify({"error": f"KeyError: {str(e)}"}), 400  # Bad Request
    except IOError as e:
        return jsonify({"error": f"IOError: {str(e)}"}), 500  # Internal Server Error


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
