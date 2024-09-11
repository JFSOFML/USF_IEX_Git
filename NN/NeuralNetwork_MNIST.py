"""Entry file for Streamlit app NN"""

import streamlit as st

# Creating sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Pytorch", "Tensorflow", "Tensorboard"])

if page == "Overview":
    st.title("MNIST Classification Project")
    st.write("Author: Julio Figueroa")
    st.divider()
    st.subheader(
        "Neural Network Implementation with PyTorch \
            and TensorFlow on the MNIST Dataset"
    )

    st.write(
        """
    The MNIST dataset remains a cornerstone in the field \
        of machine learning and computer vision education.
    Its simplicity, combined with the wealth of available \
        resources and community support,
    makes it an ideal starting point for anyone interested \
        in learning about neural networks and deep learning \
            frameworks such as PyTorch and TensorFlow.
    By working with MNIST, beginners can build a solid \
        foundation in machine learning, understand the \
            principles of neural networks,
    and gain hands-on experience with two of the most \
        popular deep learning libraries.
    """
    )

    st.write("Accuracy")
    st.divider()
    st.subheader("History of the MNIST Dataset")
    st.write(
        """
The MNIST (Modified National Institute of Standards and \
    Technology) dataset is a large collection of \
    handwritten digits that is commonly used for \
    training various image processing systems. \
    It was created by Yann LeCun, Corinna Cortes, \
    and Christopher Burges. The dataset is a subset of a \
        larger set available from NIST, and it was reprocessed \
            to create the MNIST collection.

Origin: The dataset was created by combining two of NIST's \
    databases: Special Database 1 and Special Database 3.

These databases contain binary images of handwritten digits.

Release: The MNIST dataset was released in the 1990s and \
    quickly became a benchmark for evaluating machine \
        learning models and algorithms.
"""
    )
    st.divider()
    st.subheader("Conclusion")
    st.write(
        """This project provides a comprehensive overview \
            of Neural Networks using both PyTorch and \
                TensorFlow on the MNIST dataset.
    It showcases the key steps involved in building, \
        training, and evaluating neural networks with \
            these two powerful frameworks,
    offering a hands-on approach to understanding the \
        fundamentals of deep learning."""
    )

if page == "Pytorch":
    st.header("Pytorch")
    st.write(
        """
    This project demonstrates a basic neural network for \
        classifying handwritten digits from the MNIST dataset.
    The process includes several key steps:
    """
    )

    st.subheader("1. Loading the Data")
    st.write(
        """
    The MNIST dataset is loaded using the torchvision library. \
        It consists of 60,000 training images and 10,000 \
            testing images of handwritten digits (0-9).
    """
    )

    st.subheader("2. Data Transformation")
    st.write(
        """
    The data is transformed into tensors and normalized. \
        This prepares the data for training the neural network.
    """
    )

    st.subheader("3. Creating Data Loaders")
    st.write(
        """
    Data loaders are created to feed the data to the model in \
        batches. This helps in efficient training and reduces \
            overfitting by shuffling the data.
    """
    )

    st.subheader("4. Initializing the Model")
    st.write(
        """
    A neural network model is defined with two fully connected \
        layers. The first layer has 512 neurons with ReLU \
            activation, and the second layer has 10 neurons \
                for each digit class (0-9).
    """
    )

    st.subheader("5. Setting Loss Function and Optimizer")
    st.write(
        """
    The cross-entropy loss function is used to measure the \
        performance of the model. The Adam optimizer is used \
            to update the model parameters.
    """
    )

    st.subheader("6. Training the Model")
    st.write(
        """
    The model is trained for 10 epochs. In each epoch, the \
        model's parameters are updated to minimize the loss. \
            The training loop involves the following steps:
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
    After training, the model is evaluated on the test set. \
        The average loss and accuracy are calculated and displayed.
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
    This project demonstrates a basic neural network for \
        classifying handwritten digits from the MNIST dataset.
    The process includes several key steps:
    """
    )

    st.subheader("1. Loading and Preprocessing the Data")
    st.write(
        """
    The MNIST dataset is loaded using the TensorFlow library. \
        The images are normalized by dividing the pixel values \
            by 255.0 to scale them between 0 and 1.
    """
    )

    st.subheader("2. Displaying Sample Images")
    st.write(
        """
    Five sample images from the MNIST dataset are displayed to \
        show the type of data being used. Each image is shown \
            with its corresponding label.
    """
    )
    st.image(
        "Pictures/TensorFlow3.png", caption="Sample MNIST Image", use_column_width=True
    )

    st.subheader("3. Defining and Compiling the Model")
    st.write(
        """
    A neural network model is defined with the following layers:
    - **Flatten Layer**: Converts each 28x28 image into a 1D \
        array of 784 pixels.
    - **Dense Layer**: A fully connected layer with 128 \
        neurons and ReLU activation.
    - **Dropout Layer**: A dropout layer with a 20% dropout \
        rate to prevent overfitting.
    - **Dense Layer**: The output layer with 10 neurons \
        (one for each digit class) and softmax activation.

    The model is compiled with the Adam optimizer, sparse \
        categorical cross-entropy loss, and accuracy as the \
            metric.
    """
    )

    st.subheader("4. Training the Model")
    st.write(
        """
    The model is trained for 6 epochs on the training data. \
        The training process includes adjusting the model \
            parameters to minimize the loss function.
    """
    )
    st.image(
        "Pictures/TensorFlow1.png", caption="Training Process", use_column_width=True
    )

    st.subheader("5. Evaluating the Model")
    st.write(
        """
    After training, the model is evaluated on the test data to \
        measure its performance. The test loss and accuracy are \
            computed and displayed.
    """
    )

    st.subheader("6. Visualizing Training Progress")
    st.write(
        """
    The training history, including the training loss and \
        accuracy over epochs, is plotted to visualize the \
            model's learning progress.
    """
    )
    st.image(
        "Pictures/TensorFlow2.png",
        caption="Training Loss and Accuracy",
        use_column_width=True,
    )
    st.balloons()

if page == "Tensorboard":
    st.header("Tensorboard Visualizations")
    st.write(
        "This section displays three TensorBoard \
            visualizations from the MNIST classification project."
    )

    st.image("Pictures/TensBoard_Pytorch.png", caption="TensorBoard - Pytorch")
    st.image("Pictures/TensBoard_VCurve.png", caption="TensorBoard - Validation Curve")
    st.image("Pictures/TensorBoard_1.png", caption="TensorBoard - Overview")
