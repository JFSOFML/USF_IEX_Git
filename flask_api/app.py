# Flask: Lets us create a web app
from flask import Flask, jsonify, request

# Pickle: Lets us save and load our trained model
import pickle

# NumPy: Helps with number crunching and handling arrays
import numpy as np

app = Flask(__name__)

model = pickle.load(open("Model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/", methods=["GET"])
def test():
    if request.method == "GET":
        return jsonify({"message": "Hello World"})


@app.route("/predict", methods=["POST"])
def predict():
    # Get the JSON data from the post request
    data = request.get_json()  # saves the JSON data as the variable "data"

    # extracting binary data from the JSON Which is stored in the variable name "data" starting at the first element.
    input_data = data["data"][
        0
    ]  # creating the variable data and Extracting the Data from the variable Name "data"

    # Convert to Numpy & Reshape the data to 2D for the model
    input_data = np.array(input_data).reshape(1, -1)

    # Apply the scaler
    scaled_data = scaler.transform(input_data)

    # Make prediction using the loaded model
    prediction = model.predict(scaled_data)

    # Return the prediction as a JSON response
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")  # Local Host

#     {
#     "data": [
#         {
#             "Age": 25,
#             "Sex_binary": 1,
#             "FirstClass": 0,
#             "SecondClass": 1,
#             "ThirdClass": 0
#         }
#     ]
# }
