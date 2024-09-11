"""Entry point app for streamlit"""

import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


st.header("Titanic Classification Project")
st.write(
    """
---
Author: Julio Figueroa
         
---
         

## Summarizing Data
         
We are going to run a classification machine learning algorithm 
         to find out who died and who survived. 

---
"""
)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender_submission = pd.read_csv("gender_submission.csv")

st.dataframe(train)
st.caption("This is the trained data.")
st.divider()
st.dataframe(test)
st.caption("This is the data we used to generate predictions.")
st.divider()

st.dataframe(gender_submission)
st.caption("This is the data we used to test the data accuracy.")
st.divider()

train["Sex_binary"] = train.Sex.map({"male": 0, "female": 1})
test["Sex_binary"] = test.Sex.map({"male": 0, "female": 1})

# Used pandas:Took and mergeed PassengerID and on two different
# CSVs(gender_Sub & Test) Think Inner Join SQL
test = pd.merge(test, gender_submission, on="PassengerId", how="inner")

train["FirstClass"] = train.Pclass.apply(lambda p: 1 if p == 1 else 0)
test["FirstClass"] = test.Pclass.apply(lambda p: 1 if p == 1 else 0)

train["SecondClass"] = train.Pclass.apply(lambda p: 1 if p == 2 else 0)
test["SecondClass"] = test.Pclass.apply(lambda p: 1 if p == 2 else 0)

# Created binary representation for 3nd Pclass
train["ThirdClass"] = train.Pclass.apply(lambda p: 1 if p == 3 else 0)
test["ThirdClass"] = test.Pclass.apply(lambda p: 1 if p == 3 else 0)

train["Age"].fillna(
    value=round(train["Age"].mean()), inplace=True
)  # look up .fillna function
test["Age"].fillna(value=round(test["Age"].mean()), inplace=True)
train["Age"].count()  # now we have every row accounted for.


# I want to focus on training a model on Age, Sex_binary, FirstClass,
# SecondClass, ThirdClass, "SibSp", "Parch", "Fare"
# The goal is to predict whether or not the user survived based on this.
train_features = train[
    ["Age", "Sex_binary", "FirstClass", "SecondClass", "ThirdClass"]
]  # survived = x1 + x2 + x3 + x4 + x5 + x6
train_labels = train["Survived"]  # what were trying to find. Independent Variable
test_features = test[["Age", "Sex_binary", "FirstClass", "SecondClass", "ThirdClass"]]
test_labels = test["Survived"]

X_train = train_features  # normalized training features
y_train = train_labels  # Training labels
X_test = test_features  # Normalized testing features
y_test = test_labels  # Testing labels

pipeline = Pipeline(
    [
        ("Scaler", StandardScaler()),
        ("SVC", LinearSVC()),
    ]
)

pipeline.fit(X_train, y_train)

st.write(
    """
## Model Stadardizations
         
I used SkLearn StandardScaler to normalize the data. 
         
---
"""
)
pipeline.fit(
    X_train, y_train
)  # I am fitting the output data from the Std Scaler to a linear SVC model.

linerar_SVC_predictions = pipeline.predict(
    X_test
)  # Calling the predict from the fitted model and assigning it to a variable.
linear_svc_accuracy = accuracy_score(test_labels, linerar_SVC_predictions)

st.header("Linear SVC Model Accuracy Score")
st.write(linear_svc_accuracy)
st.write(
    """
## Model Details
         
I used the SkLearn Support Vector Classifier because
          it had the highest accuracy score at 98.6%.

"""
)

with open("SVCModel_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
st.divider()
Titanic = pd.concat([train, test], ignore_index=True)

feat_labels = ["Age", "Sex_binary", "FirstClass", "SecondClass", "ThirdClass"]
