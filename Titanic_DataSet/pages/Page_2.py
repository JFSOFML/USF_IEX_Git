"""Streamlit prediction tool"""

import pickle
import streamlit as st
import pandas as pd


# Load pre-trained model and scaler
with open("SVCModel_pipeline.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)


# Define mappings for user input
sex_map = {"male": 0, "female": 1}
class_map = {"No": 0, "Yes": 1}

# User input widgets
st.header("How would you fare if you were on the Titanic?")
st.subheader("Survival Predictor Tool")


FirstClass = st.selectbox("First Class:", ["No", "Yes"])
SecondClass = st.selectbox("Second Class:", ["No", "Yes"])
ThirdClass = st.selectbox("Third Class:", ["No", "Yes"])
sex = st.selectbox("Sex:", ["male", "female"])
age = st.slider("Age:", 0, 100, 30)

# Create user input DataFrame
user_input = pd.DataFrame(
    {
        "Age": [age],
        "Sex_binary": sex_map[sex],
        "FirstClass": class_map[FirstClass],
        "SecondClass": class_map[SecondClass],
        "ThirdClass": class_map[ThirdClass],
    }
)

# Predict
if st.button("Predict"):
    # user_input_scaled = scaler .fit_transform(user_input)
    prediction = loaded_pipeline.predict(user_input)[0]

    if prediction == 1:  # Check if prediction indicates survival
        st.success("You Survived!")
    else:
        st.error("You did not survive")
