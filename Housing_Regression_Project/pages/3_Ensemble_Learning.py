"""Streamlit app for ensemble learning"""

import streamlit as st


st.header("Ensemble Learning")

# Explanation of ensemble learning techniques
st.write(
    """
### Bagging:
Bagging reduces variance by training multiple instances of a \
    model on different subsets of the data and averaging \
        their predictions.
This method helps to improve the stability and accuracy of \
    machine learning algorithms.
In this project, I used Bagging with 10 estimators to \
    enhance the model's robustness.

### Boosting:
Boosting is an iterative technique that adjusts the \
    weight of an observation based on the last classification.
It combines the performance of multiple weak learners \
    to form a strong learner, improving model accuracy \
        by focusing on the hardest to classify examples.
I applied Gradient Boosting with 100 estimators and a \
    learning rate of 1.0 to emphasize difficult cases.

### Stacking:
Stacking involves training multiple models and using \
    another model to combine their outputs.
This meta-model is trained on the predictions of base \
    models to improve generalization and model performance.
For this project, I used Linear Regression and SVR as \
    base models, with a RandomForestRegressor as the \
        meta-model.

### Voting:
Voting is an ensemble method where multiple models \
    vote on the output, and the most common prediction \
        is selected.
It can be 'hard' (majority voting) or 'soft' \
    (averaging probabilities) and is used to increase \
        prediction robustness.
I implemented Voting with Linear Regression and \
    SVR, using soft voting for probability averaging.
"""
)

st.title("Ensemble Learning on Ames Housing Dataset")

st.write(
    """
Bagging: 0.9491

Boosting: 0.8751

Stacking: 0.8315

Voting: 0.5250

"""
)