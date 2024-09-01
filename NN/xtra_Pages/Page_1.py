import streamlit as st

# Define the Streamlit app
st.title("MNIST Classification Project")

# Create sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Pytorch", "Tensorflow"])

if page == "Overview":
    st.header("Overview")
    st.write(
        """
    This is Neural Network overview using Pytorch and Tensorflow:
    """
    )
if page == "Pytorch":
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
    A neural network model is defined with two fully connected layers. The first layer has 512 neurons with ReLU activation, and the second layer has 10 neurons for each digit class (0-9).
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
    # Display the images
    st.image(
        "Pictures/MNIST_img.png", caption="Sample MNIST Image", use_column_width=True
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


if page == "Tensorflow":
    st.header("Tensorflow")
    st.write(
        """
    This project demonstrates a basic neural network for classifying handwritten digits from the MNIST dataset.
    The process includes several key steps:
    """
    )

    # Display the images
    st.image(
        "Pictures/MNIST_img.png", caption="Sample MNIST Image", use_column_width=True
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
