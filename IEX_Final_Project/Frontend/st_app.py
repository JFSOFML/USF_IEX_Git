"""Imports required for this project"""
import pickle
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import Housing_Brief as app

# Title and introduction
st.title("DS-RA IEX Final Project :hammer_and_wrench:")
st.write("By: Julio Figueroa")

st.divider()
st.header("Overview")
with st.expander("IEX Final Project Overview"):
    st.write(
        """
 This app includes projects like\
     the Titanic Survival Prediction,\
 Ames Housing Data Analysis,and MNIST Digit Classification.\
   Each project is designed to showcase different machine learning techniques:\

 ### Titanic Survival Prediction
- This section focuses on classification, predicting survival outcomes based on passenger data.\
 It demonstrates binary classification techniques\
      using logistic regression and support vector machines.\
   It highlights the use of ensemble learning methods,\
      such as Bagging and Boosting, to improve prediction accuracy.

 ### Ames Housing Data Analysis
- This section explores regression techniques, using housing data to predict sale prices.

 ### MNIST Digit Classification
- This project utilizes Convolutional Neural Networks (CNNs) to classify handwritten digits,\
      demonstrating the application of deep learning in image recognition tasks.

Navigate through the sections using the sidebar to explore different analyses and models.

***Technology Stack:***

- **Docker-Compose:** Used for orchestrating the development environment,\
      ensuring that all components work seamlessly together.
- **Flask:** Acts as the backend framework, handling API requests and\
      managing data flow between the user interface and machine learning models.
- **Streamlit:** Serves as the frontend, providing an interactive and\
      user-friendly interface for visualizing data and interacting with the models.
"""
    )
st.divider()


# Sidebar for navigation
st.sidebar.title("Navigation")
project = st.sidebar.selectbox(
    "Select a project",
    [
        "Titanic Survival Prediction",
        "Ames Housing Analysis",
        "MNIST Digit Classification",
    ],
)

# Ames Housing Analysis Section
if project == "Ames Housing Analysis":
    st.header("Ames Housing Data Analysis :house:")

    # Project Overview
    with st.expander("Project Overview"):
        st.write(
            """
        This project involves predicting housing prices in Ames, Iowa, \
            using various regression techniques. The goal is to create a model\
                  that accurately estimates the sale price of homes based on several features.\
                      Here are the main steps and processes involved:

        ### Data Preprocessing
        - **Loading Data:** The dataset was imported, focusing on specific features\
              such as 'Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air',\
                  'Total Bsmt SF', and 'SalePrice'.
        - **Feature Engineering:**
            - Converted categorical data into numerical format,\
                  such as mapping 'Central Air' to binary values.
            - Dropped any rows with missing values to maintain data integrity.

        ### Model Training and Evaluation
        - **Model Selection:** Chose `RandomForestRegressor` for its robustness\
              and ability to handle various feature interactions.
        - **Pipeline Setup:** Created a pipeline integrating data scaling (`StandardScaler`)\
              and the regression model to streamline the training process.
        - **Hyperparameter Tuning:** Utilized `RandomizedSearchCV` to optimize\
              the hyperparameters of the Random Forest model, such as the number of estimators and maximum depth.
        - **Model Evaluation:**
            - Evaluated the model using metrics like R^2, Mean Squared Error (MSE), \
                and Mean Absolute Error (MAE) to assess performance.
            - Achieved a strong R^2 score, indicating the model's ability to explain\
                  the variance in housing prices.

        ### Learning and Validation Curves
        - **Learning Curve:** Plotted learning curves to visualize the training and cross-validation scores,\
              helping to identify underfitting or overfitting.
        - **Validation Curve:** Analyzed how the model's performance varies with different hyperparameter values,\
              specifically the number of estimators in the Random Forest.

        ### Results and Insights
        - **Model Performance:** The final model showed a high level of accuracy in predicting house prices,\
              with R^2 scores indicating good model fit.
        """
        )
    st.divider()

    # Visualizations
    st.subheader("Visualizations & Tools")
    st.divider()

    column_to_filter_by = st.selectbox("Choose a column to filter by", app.df.columns)
    filter_options = st.multiselect(
        "Filter by", options=app.df[column_to_filter_by].unique()
    )

    if filter_options:
        filtered_data = app.df[app.df[column_to_filter_by].isin(filter_options)]
    else:
        filtered_data = app.df

    st.dataframe(filtered_data)
    st.write(f"{filtered_data['SalePrice'].count()} results are displayed.")
    st.divider()

    # Scatterplot Matrix
    fig = px.scatter_matrix(
        filtered_data,
        dimensions=filtered_data.columns,
        title="Scatterplot Matrix",
        width=1000,
        height=800,
    )
    fig.update_traces(
        diagonal_visible=False, showupperhalf=False, marker=dict(opacity=0.7)
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)
    st.divider()

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    corr_matrix = filtered_data.corr()
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.round(2).values,
        colorscale="Viridis",
    )
    fig.update_layout(xaxis=dict(tickangle=45))
    st.plotly_chart(fig)
    st.divider()

    # Learning and Validation Curves
    st.subheader("Learning Curve")
    st.image("Pictures/Learning_Curve_Housing.png")
    st.write(
        """
    **What It Shows**: This graph shows how the model's performance changes as the number\
          of training examples increases.
    - **Training Score (Green Line)**: How well the model fits the training data.
    - **Cross-Validation Score (Red Line)**: How well the model performs on unseen data.
    - **Interpretation**: The model performs well on the training data, but there is a gap\
          between the training and cross-validation scores, indicating some overfitting.\
              The cross-validation score improves with more training data but starts to plateau,\
                  suggesting that adding more data might not significantly improve performance.
    """
    )

    st.divider()

    st.subheader("KMeans++ Elbow Plot")
    st.image("Pictures/K_means++.png")
    st.write(
        """
    **What It Shows**: This graph helps determine the optimal number of clusters for K-Means clustering.
    - **Distortion**: The sum of squared distances from each point to its assigned cluster center.
    - **Interpretation**: The "elbow" point, where the distortion decreases more slowly,\
          suggests the optimal number of clusters. Here, it appears around 3 or 4 clusters.
    """
    )
    st.divider()

    # MAE and R^2 Explanation
    st.subheader("Model Performance Metrics")
    st.write(
        """

    **MAE Explanation**:
    - on average, our predicted house prices are off by $21,000 from the actual prices.

    **R^2 Score**:
    - **Explanation**: An R^2 score of 0.86 means that 86% of the variability in house prices\
          can be explained by the model. This is a strong indication that the model is performing well.
    """
    )

    # Predictions
    st.header("Predictions")

    overall_qual = st.select_slider(
        "Overall Quality (1-10):", options=range(1, 11), value=5
    )
    overall_cond = st.select_slider(
        "Overall Condition (1-10):", options=range(1, 11), value=5
    )
    gr_liv_area = st.slider("Above Grade Living Area (sq. ft):", 0, 5000, 1500)
    Central_Air = st.selectbox("Central Air", ["Yes", "No"])
    Central_Air_Binary = {"Yes": 1, "No": 0}
    total_bsmt_sf = st.slider("Total Basement SF (sq. ft.):", 0, 3000, 1500)

    Input ={
            "Overall Qual": [overall_qual],
            "Overall Cond": [overall_cond],
            "Total Bsmt SF": [total_bsmt_sf],
            "Central Air": Central_Air_Binary[Central_Air],
            "Gr Liv Area": [gr_liv_area],
        }

    if st.button("Predict Sale Price"):
        response = requests.post(
            "http://fl_container:5000/predict_housing", json=Input, timeout=15
        )
        result = response.json()
        result_df = pd.DataFrame([result])
        prediction = result_df["price"][0]
        st.subheader("Predicted Sale Price")
        st.write(f"${prediction:,.2f}")
    st.divider()

    # Ensemble Learning Explanation
    st.header("Ensemble Learning Techniques")
    with st.expander("Learn more about ensemble learning techniques"):
        st.write(
            """
        ### Bagging:
        Bagging reduces variance by training multiple instances of a model\
              on different subsets of the data and averaging their predictions.

        ### Boosting:
        Boosting is an iterative technique that adjusts the weight of an observation based on the last classification.

        ### Stacking:
        Stacking involves training multiple models and using another model to combine their outputs.

        ### Voting:
        Voting is an ensemble method where multiple models vote on the output, and the most common prediction is selected.
        """
        )

    st.title("Ensemble Learning Results on Ames Housing Dataset")
    st.write(
        """
    - **Bagging:** 0.9491
    - **Boosting:** 0.8751
    - **Stacking:** 0.8315
    - **Voting:** 0.5250
    """
    )

    st.divider()


# Titanic Survival Prediction Section
if project == "Titanic Survival Prediction":
    st.header("Titanic Survival Prediction :ship:")
    st.write(
        "This section will cover the analysis and predictions based on the Titanic dataset."
    )

    with st.expander("Titanic Overview"):
        st.write(
            """
        This project focuses on predicting the survival of passengers on the Titanic using machine learning techniques.\
              Below are the key steps and processes involved:

        ### Data Preprocessing & Exploratory Data Analysis (EDA)
        - **Loading Data:** Imported the train, test, and gender submission datasets.
        - **Feature Engineering:**
            - Converted the 'Sex' column into a binary format: 0 for male and 1 for female.
            - Created binary indicators for passenger classes: FirstClass, SecondClass, and ThirdClass.
        - **Handling Missing Values:** Filled missing 'Age' values with the mean age from the dataset.
        - **Data Merging:** Merged test data with the gender submission data on 'PassengerId' to create a unified test set.


        - **Visualization:** Used various visualization techniques to explore data distributions and correlations,\
              helping to understand the underlying patterns and relationships in the data.

        """
        )
        st.image(
            "Pictures/Correlation_Matrix_Titanic.png",
            caption="Correlation Matrix",
            use_column_width=True,
        )

        st.write(
            """
        ### Model Training
        - **Feature Selection:** Selected features such as 'Age', 'Sex_binary', 'FirstClass', 'SecondClass', 'ThirdClass', 'SibSp', 'Parch', and 'Fare' for model training.
        - **Data Normalization:** Applied `StandardScaler` to normalize the features, ensuring that each feature contributed equally to the model.
        - **Model Implementation:** Used a pipeline to streamline the process, integrating data normalization and the `LinearSVC` model.
        - **Training:** Fitted the model on the training data to learn the patterns associated with survival.

        ### Model Evaluation
        - **Prediction:** Used the trained model to make predictions on the test data.
        - **Accuracy Assessment:** Evaluated the model's performance using accuracy scores.

        ### Results
        - **Model Accuracy:**
          - Logistic Regression Accuracy: 85%
          - Decision Tree Accuracy: 95.7%
          - Random Forest Accuracy: 93.1%
          - **SVC Accuracy (Standard Scaler): 98.6%**

        """
        )

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
    user_input = {
            "Age": [age],
            "Sex_binary": sex_map[sex],
            "FirstClass": class_map[FirstClass],
            "SecondClass": class_map[SecondClass],
            "ThirdClass": class_map[ThirdClass],
        }

    # Predict
    if st.button("Predict"):
        response= requests.post(
            "http://fl_container:5000/predict_titanic", json=user_input, timeout=11
        )
        result= response.json()
        result_df=pd.DataFrame([result])
        #survival_prob = result_df["survival_prob"][0]
        prediction = result_df["Survived"][0]


        if prediction == 1:  # Check if prediction indicates survival
            st.success("You Survived!")
        else:
            st.error("You did not survive")

    st.divider()

    st.header("Ensemble Learning on Titanic Dataset")

    with st.expander("Detailed Explanation of Ensemble Learning Techniques"):
        st.subheader("Step 1: Imports")
        st.image("Pictures/Ensemble1.png")
        st.write(
            """
        **Why?**: We import necessary libraries and functions to handle data, preprocess it, and build and evaluate machine learning models.
        """
        )

        st.subheader("Step 2: Load Data")
        st.image("Pictures/Ensemble2.png")
        st.write(
            """
        **Why?**: Load the Titanic dataset from CSV files into pandas DataFrames for processing and analysis.
        """
        )

        st.subheader("Step 3: Fill Missing Values and Convert Categorical Data")
        st.image("Pictures/Ensemble3.png")
        st.write(
            """
        **Why?**: Fill missing values in 'Age' and 'Fare' with the mean values to ensure no gaps in the data. Convert the 'Sex' column to a binary format (0 for male, 1 for female) to use it in machine learning models.
        """
        )

        st.subheader("Step 4: Assign Features and Labels")
        st.image("Pictures/Ensemble4.png")
        st.write(
            """
        **Why?**: Select relevant features for training the models and set up the labels (target variable) for prediction. This separates the input variables (features) from the output variable (label).
        """
        )

        st.subheader("Step 5: Standardize Features")
        st.image("Pictures/Ensemble5.png")
        st.write(
            """
        **Why?**: Standardize the features to ensure they all have the same scale. This helps many machine learning algorithms perform better.
        """
        )

        st.subheader("Step 6: Define the Models")
        st.image("Pictures/Ensemble6.png")
        st.write(
            """
        **Why?**: Define multiple ensemble models:
        - **Bagging**: Combines multiple decision trees to reduce overfitting.
        - **Boosting**: Combines weak learners (shallow trees) to create a strong learner.
        - **Stacking**: Uses multiple models and combines their outputs with a final estimator.
        - **Voting**: Combines predictions from different models and takes a majority vote.
        """
        )

        st.subheader("Step 7: Train and Evaluate Each Model")
        st.image("Pictures/Ensemble7.png")
        st.write(
            """
        **Why?**: Train each model on the training data and evaluate their performance on the test data. Print the accuracy to compare how well each model predicts survival.
        """
        )

    st.subheader("Ensemble Results:")
    st.write(
        """
    Bagging Model Accuracy: 83.5%

    Boosting Model Accuracy: 92.1%

    Stacking Model Accuracy: 84.7%

    Voting Model Accuracy: 93.5%
    """
    )


# MNIST Digit Classification Section
if project == "MNIST Digit Classification":
    st.divider()

     # Sidebar for MNIST navigation
    st.sidebar.title("MNIST Navigation")
    mnist_page = st.sidebar.radio("Go to", ["Overview", "Pytorch", "Tensorflow"])

    if mnist_page == "Overview":
        st.title("MNIST Classification Project")
        st.write("Author: Julio Figueroa")
        st.image("Pictures/MNIST_1.png")
        st.divider()

        st.subheader("History of the MNIST Dataset")
        st.write(
            """
        The MNIST (Modified National Institute of Standards and Technology) dataset is a large collection of handwritten digits that is commonly used for training various image processing systems. It was created by Yann LeCun, Corinna Cortes, and Christopher Burges. The dataset is a subset of a larger set available from NIST, and it was reprocessed to create the MNIST collection.
        """
        )

        st.divider()
        st.subheader("Conclusion")
        st.write(
            """This project provides the steps of setting up a CNN using both PyTorch and TensorFlow on the MNIST dataset.
        It showcases the key steps involved in building, training, and evaluating neural networks with these two powerful frameworks,
        offering a hands-on approach to understanding the fundamentals of deep learning."""
        )

    elif mnist_page == "Pytorch":
        st.header("Pytorch")
        st.write(
            """
        This project demonstrates a basic neural network for classifying handwritten digits from the MNIST dataset.
        The process includes several key steps:
        """
        )

        st.subheader("1. Loading the Data")
        st.write(
            """
        The MNIST dataset is loaded using the torchvision library. It consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9).
        """
        )

        st.subheader("2. Data Transformation")
        st.write(
            """
        The data is transformed into tensors and normalized. This prepares the data for training the neural network.
        """
        )

        st.subheader("3. Creating Data Loaders")
        st.write(
            """
        Data loaders are created to feed the data to the model in batches. This helps in efficient training and reduces overfitting by shuffling the data.
        """
        )

        st.subheader("4. Initializing the Model")
        st.write(
            """
        The neural network model is defined with two fully connected layers. The first layer has 512 neurons with ReLU activation, and the second layer has 10 neurons for each digit class (0-9).
        """
        )

        st.subheader("5. Setting Loss Function and Optimizer")
        st.write(
            """
        The cross-entropy loss function is used to measure the performance of the model. The Adam optimizer is used to update the model parameters.
        """
        )

        st.subheader("6. Training the Model")
        st.write(
            """
        The model is trained for 10 epochs. In each epoch, the model's parameters are updated to minimize the loss. The training loop involves the following steps:
        - Zero the gradients
        - Perform a forward pass to compute predictions
        - Compute the loss
        - Perform a backward pass to compute gradients
        - Update the model parameters
        """
        )

        st.header("Training Loop")
        st.write(
            """
        Epoch 0: Loss: 0.1448

        Epoch 1: Loss: 0.0311

        Epoch 2: Loss: 0.0391

        Epoch 3: Loss: 0.0193

        Epoch 4: Loss: 0.0075

        Epoch 5: Loss: 0.0761

        Epoch 6: Loss: 0.0200

        Epoch 7: Loss: 0.0181

        Epoch 8: Loss: 0.0001

        Epoch 9: Loss: 0.0005
        """
        )

        st.subheader("7. Evaluating the Model")
        st.write(
            """
        After training, the model is evaluated on the test set. The average loss and accuracy are calculated and displayed.
        """
        )
        st.divider()
        st.image(
            "Pictures/MNIST_img.png",
            caption="Sample MNIST Image",
            use_column_width=True,
        )
        st.image(
            "Pictures/Ptorch_LC_Loss.png",
            caption="Learning Curve - Loss",
            use_column_width=True,
        )
        st.image(
            "Pictures/Ptorch_LC_Accuracy.png",
            caption="Learning Curve - Accuracy",
            use_column_width=True,
        )

    elif mnist_page == "Tensorflow":
        st.header("Tensorflow")
        st.write(
            """
        This project demonstrates a basic neural network for classifying handwritten digits from the MNIST dataset.
        The process includes several key steps:
        """
        )

        st.subheader("1. Loading and Preprocessing the Data")
        st.write(
            """
        The MNIST dataset is loaded using the TensorFlow library. The images are normalized by dividing the pixel values by 255.0 to scale them between 0 and 1.
        """
        )

        st.subheader("2. Displaying Sample Images")
        st.write(
            """
        Five sample images from the MNIST dataset are displayed to show the type of data being used. Each image is shown with its corresponding label.
        """
        )
        st.image(
            "Pictures/TensorFlow3.png",
            caption="Sample MNIST Image",
            use_column_width=True,
        )

        st.subheader("3. Defining and Compiling the Model")
        st.write(
            """
        A neural network model is defined with the following layers:
        - **Flatten Layer**: Converts each 28x28 image into a 1D array of 784 pixels.
        - **Dense Layer**: A fully connected layer with 128 neurons and ReLU activation.
        - **Dropout Layer**: A dropout layer with a 20% dropout rate to prevent overfitting.
        - **Dense Layer**: The output layer with 10 neurons (one for each digit class) and softmax activation.

        The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the metric.
        """
        )

        st.subheader("4. Training the Model")
        st.write(
            """
        The model is trained for 6 epochs on the training data. The training process includes adjusting the model parameters to minimize the loss function.
        """
        )
        st.image(
            "Pictures/TensorFlow1.png",
            caption="Training Process",
            use_column_width=True,
        )

        st.subheader("5. Evaluating the Model")
        st.write(
            """
        After training, the model is evaluated on the test data to measure its performance. The test loss and accuracy are computed and displayed.
        """
        )

        st.subheader("6. Visualizing Training Progress")
        st.write(
            """
        The training history, including the training loss and accuracy over epochs, is plotted to visualize the model's learning progress.
        """
        )
        st.image(
            "Pictures/TensorFlow2.png",
            caption="Training Loss and Accuracy",
            use_column_width=True,
        )
        st.image(
            "Pictures/epoch_Tensor_Training1.png",
            caption="Training Loss and Accuracy",
            use_column_width=True,
        )
