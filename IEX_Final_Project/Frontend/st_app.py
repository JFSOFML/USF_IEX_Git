import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.figure_factory as ff
import Housing_Brief as app

# Title and introduction
st.title("DS-RA IEX Final Project :hammer_and_wrench:")
st.write("By: Julio Figueroa")

st.divider()
st.header("Overview")
with st.expander("IEX Final Project Overview"): 
    st.write("""
This app includes projects like the Titanic Survival Prediction, Ames Housing Data Analysis, and MNIST Digit Classification. Each project is designed to showcase different machine learning techniques:

 ### Titanic Survival Prediction 
- This section focuses on classification, predicting survival outcomes based on passenger data. It demonstrates binary classification techniques using logistic regression and support vector machines.

 ### Ames Housing Data Analysis
- This section explores regression techniques, using housing data to predict sale prices. It highlights the use of ensemble learning methods, such as Bagging and Boosting, to improve prediction accuracy.

 ### MNIST Digit Classification
- This project utilizes Convolutional Neural Networks (CNNs) to classify handwritten digits, demonstrating the application of deep learning in image recognition tasks.

Navigate through the sections using the sidebar to explore different analyses and models.

***Technology Stack:***
             
- **Docker-Compose:** Used for orchestrating the development environment, ensuring that all components work seamlessly together.
- **Flask:** Acts as the backend framework, handling API requests and managing data flow between the user interface and machine learning models.
- **Streamlit:** Serves as the frontend, providing an interactive and user-friendly interface for visualizing data and interacting with the models.
""")
st.divider()


# Sidebar for navigation
st.sidebar.title("Navigation")
project = st.sidebar.selectbox("Select a project", ["Titanic Survival Prediction", "Ames Housing Analysis", "MNIST Digit Classification"])

# Ames Housing Analysis Section
if project == "Ames Housing Analysis":
    st.header("Ames Housing Data Analysis :house:")
    
    # Visualizations
    st.subheader("Visualizations")
    st.divider()

    column_to_filter_by = st.selectbox("Choose a column to filter by", app.df.columns)
    filter_options = st.multiselect("Filter by", options=app.df[column_to_filter_by].unique())

    if filter_options:
        filtered_data = app.df[app.df[column_to_filter_by].isin(filter_options)]
    else:
        filtered_data = app.df

    st.dataframe(filtered_data)
    st.write(f"{filtered_data['SalePrice'].count()} results are displayed.")
    st.divider()

    # Scatterplot Matrix
    fig = px.scatter_matrix(filtered_data, dimensions=filtered_data.columns, title='Scatterplot Matrix', width=1000, height=800)
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(opacity=0.7))
    fig.update_xaxes(tickangle=45) 
    st.plotly_chart(fig)
    st.divider()

    # Correlation Matrix
    st.subheader('Correlation Matrix')
    corr_matrix = filtered_data.corr()
    fig = ff.create_annotated_heatmap(z=corr_matrix.values, x=list(corr_matrix.columns), y=list(corr_matrix.index), annotation_text=corr_matrix.round(2).values, colorscale='Viridis')
    fig.update_layout(xaxis=dict(tickangle=45))
    st.plotly_chart(fig)
    st.divider()

    # Learning and Validation Curves
    st.subheader("Learning Curve")
    st.image("Pictures/Learning_Curve_Housing.png")
    st.divider()

    st.subheader("Validation Curve")
    st.image("Pictures/Housing_Validation_Curve.png")
    st.write("R2 Score = 0.8412 ± 0.0607")
    st.divider()

    st.subheader("KMeans++ Elbow Plot")
    st.image("Pictures/K_means++.png")
    st.divider()

    # Predictions
    st.header("Predictions")

    with open('Scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
    with open('forest.pkl', 'rb') as f:
        model = pickle.load(f)

    overall_qual = st.select_slider('Overall Quality (1-10):', options=range(1, 11), value=5)
    overall_cond = st.select_slider('Overall Condition (1-10):', options=range(1, 11), value=5)
    gr_liv_area = st.slider('Above Grade Living Area (sq. ft):', 0, 5000, 1500)
    Central_Air = st.selectbox('Central Air',["Yes","No"])
    Central_Air_Binary = {'Yes':1, 'No':0}
    total_bsmt_sf = st.slider('Total Basement SF (sq. ft.):', 0, 3000, 1500)

    Input = pd.DataFrame({
        'Overall Qual': [overall_qual],
        'Overall Cond': [overall_cond],
        'Total Bsmt SF': [total_bsmt_sf],
        'Central Air': Central_Air_Binary[Central_Air],
        'Gr Liv Area': [gr_liv_area]
    })

    if st.button("Predict Sale Price"):
        user_input_scaled = sc.transform(Input)
        prediction = model.predict(user_input_scaled)[0]
        st.subheader("Predicted Sale Price")
        st.write(f"${prediction:,.2f}")
    st.divider()

    # Ensemble Learning Explanation
    st.header("Ensemble Learning Techniques")
    with st.expander("Learn more about ensemble learning techniques"):
        st.write("""
        ### Bagging: 
        Bagging reduces variance by training multiple instances of a model on different subsets of the data and averaging their predictions.

        ### Boosting: 
        Boosting is an iterative technique that adjusts the weight of an observation based on the last classification.

        ### Stacking: 
        Stacking involves training multiple models and using another model to combine their outputs.

        ### Voting: 
        Voting is an ensemble method where multiple models vote on the output, and the most common prediction is selected.
        """)

    st.title("Ensemble Learning Results on Ames Housing Dataset")
    st.write("""
    - **Bagging:** 0.9491
    - **Boosting:** 0.8751
    - **Stacking:** 0.8315
    - **Voting:** 0.5250
    """)

# Titanic Survival Prediction Section
if project == "Titanic Survival Prediction":
    st.header("Titanic Survival Prediction :ship:")
    st.write("This section will cover the analysis and predictions based on the Titanic dataset.")
    
    with st.expander("Titanic Overview"):
        st.write("""
        This project focuses on predicting the survival of passengers on the Titanic using machine learning techniques. Below are the key steps and processes involved:

        ### Data Preprocessing
        - **Loading Data:** Imported the train, test, and gender submission datasets.
        - **Feature Engineering:**
            - Converted the 'Sex' column into a binary format: 0 for male and 1 for female.
            - Created binary indicators for passenger classes: FirstClass, SecondClass, and ThirdClass.
        - **Handling Missing Values:** Filled missing 'Age' values with the mean age from the dataset.
        - **Data Merging:** Merged test data with the gender submission data on 'PassengerId' to create a unified test set.

        ### Exploratory Data Analysis (EDA)
        - **Visualization:** Used various visualization techniques to explore data distributions and correlations, helping to understand the underlying patterns and relationships in the data.

        ### Model Training
        - **Feature Selection:** Selected features such as 'Age', 'Sex_binary', 'FirstClass', 'SecondClass', 'ThirdClass', 'SibSp', 'Parch', and 'Fare' for model training.
        - **Data Normalization:** Applied `StandardScaler` to normalize the features, ensuring that each feature contributed equally to the model.
        - **Model Implementation:** Used a pipeline to streamline the process, integrating data normalization and the `LinearSVC` model.
        - **Training:** Fitted the model on the training data to learn the patterns associated with survival.

        ### Model Evaluation
        - **Prediction:** Used the trained model to make predictions on the test data.
        - **Accuracy Assessment:** Evaluated the model's performance using accuracy scores, achieving notable accuracy with the `LinearSVC` model.

        ### Results
        - **Model Accuracy:** The `LinearSVC` model achieved an accuracy score of approximately 98.6%, indicating a strong predictive performance.
        - **Feature Importance:** While not detailed here, further analysis can include identifying which features most significantly impacted survival predictions.

        This project demonstrates the application of data preprocessing, feature engineering, and machine learning techniques to solve a real-world classification problem.
        """)

    # Define mappings for user input
    sex_map = {'male': 0, 'female': 1}
    class_map = {'No': 0, 'Yes': 1}

    # User input widgets
    st.header("How would you fare if you were on the Titanic?")
    st.subheader("Survival Predictor Tool")

    FirstClass = st.selectbox('First Class:', ['No', 'Yes'])
    SecondClass = st.selectbox('Second Class:', ['No', 'Yes'])
    ThirdClass = st.selectbox('Third Class:', ['No', 'Yes'])
    sex = st.selectbox('Sex:', ['male', 'female'])
    age = st.slider('Age:', 0, 100, 30)

    # Create user input DataFrame
    user_input = pd.DataFrame({
        'Age': [age], 
        'Sex_binary': sex_map[sex], 
        'FirstClass': class_map[FirstClass], 
        'SecondClass': class_map[SecondClass], 
        'ThirdClass': class_map[ThirdClass]
    })

    loaded_pipeline = pickle.load(open('SVCModel_pipeline.pkl','rb'))
    # Predict 
    if st.button("Predict"):
        prediction = loaded_pipeline.predict(user_input)[0]

        if prediction == 1:  # Check if prediction indicates survival
            st.success("You Survived!") 
        else:
            st.error("You did not survive")

    st.divider()


    st.header("Ensemble Learning")
    with st.expander("Detailed Explanation of Ensemble Learning Techniques"):
        st.write("""
        ### Bagging: 
        Bagging reduces variance by training multiple instances of a model on different subsets of the data and averaging their predictions.
        This method helps to improve the stability and accuracy of machine learning algorithms.
        In this project, I used Bagging with 100 estimators to enhance the model's robustness.
        
        ### Boosting: 
        Boosting is an iterative technique that adjusts the weight of an observation based on the last classification.
        It combines the performance of multiple weak learners to form a strong learner, improving model accuracy by focusing on the hardest to classify examples.
        I applied Gradient Boosting with 100 estimators and a learning rate of 1.0 to emphasize difficult cases.
        
        ### Stacking: 
        Stacking involves training multiple models and using another model to combine their outputs.
        This meta-model is trained on the predictions of base models to improve generalization and model performance.
        For this project, I used Logistic Regression and SVC as base models, with a RandomForestClassifier as the meta-model.
        
        ### Voting: 
        Voting is an ensemble method where multiple models vote on the output, and the most common prediction is selected.
        It can be 'hard' (majority voting) or 'soft' (averaging probabilities) and is used to increase prediction robustness.
        I implemented Voting with Logistic Regression, RandomForestClassifier, and SVC, using soft voting for probability averaging.
        """)

    st.title("Ensemble Learning on Titanic Dataset")
    st.write("""
    Bagging Model Accuracy: 0.8349
            
    Boosting Model Accuracy: 0.9211
            
    Stacking Model Accuracy: 0.8469
            
    Voting Model Accuracy: 0.9354
    """)

# MNIST Digit Classification Section
if project == "MNIST Digit Classification":
    st.header("MNIST Digit Classification")
    st.write("This section will showcase the digit classification using the MNIST dataset.")
    st.divider()

    # Sidebar for MNIST navigation
    st.sidebar.title("MNIST Navigation")
    mnist_page = st.sidebar.radio("Go to", ["Overview", "Pytorch","Tensorflow"])

    if mnist_page == "Overview":
        st.title("MNIST Classification Project")
        st.write("Author: Julio Figueroa")
        st.divider()
        st.subheader("Neural Network Implementation with PyTorch and TensorFlow on the MNIST Dataset")
        
        st.write("""
        The MNIST dataset remains a cornerstone in the field of machine learning and computer vision education. 
        Its simplicity, combined with the wealth of available resources and community support,
        makes it an ideal starting point for anyone interested in learning about neural networks and deep learning frameworks such as PyTorch and TensorFlow.
        By working with MNIST, beginners can build a solid foundation in machine learning, understand the principles of neural networks,
        and gain hands-on experience with two of the most popular deep learning libraries.
        """)

        st.divider()
        st.subheader("History of the MNIST Dataset")
        st.write("""
        The MNIST (Modified National Institute of Standards and Technology) dataset is a large collection of handwritten digits that is commonly used for training various image processing systems. It was created by Yann LeCun, Corinna Cortes, and Christopher Burges. The dataset is a subset of a larger set available from NIST, and it was reprocessed to create the MNIST collection.

        Origin: The dataset was created by combining two of NIST's databases: Special Database 1 and Special Database 3.

        These databases contain binary images of handwritten digits.

        Release: The MNIST dataset was released in the 1990s and quickly became a benchmark for evaluating machine learning models and algorithms.
        """)
        st.divider()
        st.subheader("Conclusion")
        st.write("""This project provides a comprehensive overview of Neural Networks using both PyTorch and TensorFlow on the MNIST dataset.
        It showcases the key steps involved in building, training, and evaluating neural networks with these two powerful frameworks,
        offering a hands-on approach to understanding the fundamentals of deep learning.""")

    elif mnist_page == "Pytorch":
        st.header("Pytorch")
        st.write("""
        This project demonstrates a basic neural network for classifying handwritten digits from the MNIST dataset.
        The process includes several key steps:
        """)
        
        st.subheader("1. Loading the Data")
        st.write("""
        The MNIST dataset is loaded using the torchvision library. It consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9).
        """)

        st.subheader("2. Data Transformation")
        st.write("""
        The data is transformed into tensors and normalized. This prepares the data for training the neural network.
        """)

        st.subheader("3. Creating Data Loaders")
        st.write("""
        Data loaders are created to feed the data to the model in batches. This helps in efficient training and reduces overfitting by shuffling the data.
        """)

        st.subheader("4. Initializing the Model")
        st.write("""
        A neural network model is defined with two fully connected layers. The first layer has 512 neurons with ReLU activation, and the second layer has 10 neurons for each digit class (0-9).
        """)

        st.subheader("5. Setting Loss Function and Optimizer")
        st.write("""
        The cross-entropy loss function is used to measure the performance of the model. The Adam optimizer is used to update the model parameters.
        """)

        st.subheader("6. Training the Model")
        st.write("""
        The model is trained for 10 epochs. In each epoch, the model's parameters are updated to minimize the loss. The training loop involves the following steps:
        - Zero the gradients
        - Perform a forward pass to compute predictions
        - Compute the loss
        - Perform a backward pass to compute gradients
        - Update the model parameters
        """)

        st.header("Training Loop")
        st.write("""
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
        """)

        st.subheader("7. Evaluating the Model")
        st.write("""
        After training, the model is evaluated on the test set. The average loss and accuracy are calculated and displayed.
        """)
        st.divider()
        st.image("Pictures/MNIST_img.png", caption='Sample MNIST Image', use_column_width=True)
        st.image("Pictures/Ptorch_LC_Loss.png", caption='Learning Curve - Loss', use_column_width=True)
        st.image("Pictures/Ptorch_LC_Accuracy.png", caption='Learning Curve - Accuracy', use_column_width=True)


    elif mnist_page == "Tensorflow":
        st.header("Tensorflow")
        st.write("""
        This project demonstrates a basic neural network for classifying handwritten digits from the MNIST dataset.
        The process includes several key steps:
        """)

        st.subheader("1. Loading and Preprocessing the Data")
        st.write("""
        The MNIST dataset is loaded using the TensorFlow library. The images are normalized by dividing the pixel values by 255.0 to scale them between 0 and 1.
        """)

        st.subheader("2. Displaying Sample Images")
        st.write("""
        Five sample images from the MNIST dataset are displayed to show the type of data being used. Each image is shown with its corresponding label.
        """)
        st.image("Pictures/TensorFlow3.png", caption='Sample MNIST Image', use_column_width=True)

        st.subheader("3. Defining and Compiling the Model")
        st.write("""
        A neural network model is defined with the following layers:
        - **Flatten Layer**: Converts each 28x28 image into a 1D array of 784 pixels.
        - **Dense Layer**: A fully connected layer with 128 neurons and ReLU activation.
        - **Dropout Layer**: A dropout layer with a 20% dropout rate to prevent overfitting.
        - **Dense Layer**: The output layer with 10 neurons (one for each digit class) and softmax activation.
        
        The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the metric.
        """)

        st.subheader("4. Training the Model")
        st.write("""
        The model is trained for 6 epochs on the training data. The training process includes adjusting the model parameters to minimize the loss function.
        """)
        st.image("Pictures/TensorFlow1.png", caption='Training Process', use_column_width=True)

        st.subheader("5. Evaluating the Model")
        st.write("""
        After training, the model is evaluated on the test data to measure its performance. The test loss and accuracy are computed and displayed.
        """)

        st.subheader("6. Visualizing Training Progress")
        st.write("""
        The training history, including the training loss and accuracy over epochs, is plotted to visualize the model's learning progress.
        """)
        st.image("Pictures/TensorFlow2.png", caption='Training Loss and Accuracy', use_column_width=True)
