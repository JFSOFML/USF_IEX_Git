from flask import Flask, jsonify, request 
import pickle

app = Flask (__name__)

model = pickle.load(open("SVCModel_pipeline.pkl", "rb"))

@app.route("/", methods=["GET"])
def test():
    if request.method == "GET": 
        return jsonify({"message": "Hello World"})
    
if __name__ == "__main__": 
    app.run(debug=True, host="0.0.0.0")#Local Host