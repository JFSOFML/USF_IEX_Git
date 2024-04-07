# import streamlit as st

# st.header("Prediction Manipulations")
# import app
# import streamlit as st
# import pandas as pd
# import pickle
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.calibration import CalibratedClassifierCV
# min_max_scaler = MinMaxScaler()
# st.header("How would you fare if you were on the Titanic?")

# st.subheader("Survival Predictor Tool")

# sex_map = {'male': 0, 'female': 1}
# class_map = {'No': 0, 'Yes': 1}

# # feat_labels = ['Age', 'Sex_binary', 'FirstClass', 'SecondClass', 'ThirdClass']

# # pclass = st.selectbox('Passenger Class:', [1, 2, 3])
# FirstClass = st.selectbox('First Class:', ['No', 'Yes'])
# SecondClass= st.selectbox('Second Class:', ['No', 'Yes'])
# ThirdClass = st.selectbox('Third Class:', ['No', 'Yes'])
# sex = st.selectbox('Sex:', ['male', 'female'])
# age = st.slider('Age:', 0, 100, 30)


# user_input = pd.DataFrame({
#     # 'Pclass': [pclass],
#     'Age': [age],
#     'Sex': sex_map[sex],
#     'First Class': class_map[FirstClass],
#     'Second Class': class_map[SecondClass],
#     'Third Class': class_map[ThirdClass]
# })

# with open('SVCModel.pkl', 'rb') as f:
#     model = pickle.load(f)

# if st.button("Predict"):
#     user_input_scaled = min_max_scaler.fit_transform(user_input)

#     prediction = model.predict(user_input_scaled)[0]
#     # survival_prob = model.predict_proba(user_input_scaled)[0][1] 
#     # not_survived_prob = 1 - survival_prob
#     # Assuming your LinearSVC model is already trained and named `model`
#     calibrated_svc = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
#     calibrated_svc.fit(user_input_scaled, app.train_labels)  # You need to provide the training data labels here as `y_train`

#     # Then you can use predict_proba on calibrated_svc
#     survival_prob = calibrated_svc.predict_proba(user_input_scaled)[0][1]
#     not_survived_prob = 1 - survival_prob
#     # Display the outcome
#     if not_survived_prob != 0:
#         st.success("You Survived!")

#     else:
#         st.error("You did not survive")


import streamlit as st
import pandas as pd
import pickle 
from sklearn.preprocessing import StandardScaler

# Load pre-trained model and scaler
with open('SVCModel.pkl', 'rb') as f:
    model = pickle.load(f)
scaler = StandardScaler()

# Define mappings for user input
sex_map = {'male': 0, 'female': 1}
class_map = {'No': 0, 'Yes': 1}

# User input widgets
st.header("How would you fare if you were on the Titanic?")
st.subheader("Survival Predictor Tool")


FirstClass = st.selectbox('First Class:', ['No', 'Yes'])
SecondClass= st.selectbox('Second Class:', ['No', 'Yes'])
ThirdClass = st.selectbox('Third Class:', ['No', 'Yes'])
sex = st.selectbox('Sex:', ['male', 'female'])
age = st.slider('Age:', 0, 100, 30)

# Create user input DataFrame
user_input = pd.DataFrame({
    'Age': [age], 
    'Sex': sex_map[sex], 
    'First Class': class_map[FirstClass], 
    'Second Class': class_map[SecondClass], 
    'Third Class': class_map[ThirdClass]
})

# Predict 
if st.button("Predict"):
    user_input_scaled = scaler .fit_transform(user_input)
    prediction = model.predict(user_input_scaled)[0]

    if prediction == 1:  # Check if prediction indicates survival
        st.success("You Survived!") 
    else:
        st.error("You did not survive")


