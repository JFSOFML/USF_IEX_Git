from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

def execute_query(query):
    try:
        conn = sqlite3.connect("/app/titanic.db")
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        data = cursor.fetchall()
        conn.close()
        return {"columns": columns, "data": data}
    except Exception as e:
        return {"error": str(e)}

@app.route("/query", methods=["POST"])
def query():
    try:
        # GET: the JSON data from the POST request
        request_data = request.json
        
        # EXTRACT: the query from the JSON data
        query = request_data.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # EXECUTE: the query
        result = execute_query(query)
        
        # RETURN: the result as a JSON response
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
