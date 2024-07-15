from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

# Logic for executing query in SQLite | Come back to this in flask : Two Functions 
def execute_query(query):   # sql stuff
    pass #return {"columns": columns, "data": data}

@app.route("/Qry", methods=["POST"])
def query():   #Flask related
    pass

if __name__== "__main__":
    app.run(host="0.0.0.0", debug=True)