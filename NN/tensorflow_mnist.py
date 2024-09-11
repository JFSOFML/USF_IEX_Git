"""Convert from ipynb"""

import datetime
import keras
import matplotlib.pyplot as plt

# %load_ext tensorboard


# Clear any logs from previous runs
# rm -rf ./logs/


# Load Mnist data
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# Display 5 images
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(x_train[i], cmap="gray")  # Use grayscale for better visualization
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")  # Removes the axis ticks if set to off
plt.show()


def create_model():
    """Create the model"""
    return keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(
                0.2
            ),  # randomly turns off 20% of input to zero to prevent overfitting
            keras.layers.Dense(
                10, activation="softmax"
            ),  # like hard & Soft voting in Ensemble learning
        ]
    )


model = create_model()


model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


# model.fit(x_train, y_train, epochs=6)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    x=x_train,
    y=y_train,
    epochs=6,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
)


test_loss, test_acc = model.evaluate(x_test, y_test)

# Printed results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


history = model.fit(x_train, y_train, epochs=6)


train_loss = history.history["loss"]
train_acc = history.history["accuracy"]


plt.figure(figsize=(12, 7))
plt.plot(train_loss, color="purple", label="Training Loss")  # Changed color to purple
plt.title("Training Loss Over Epochs", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", linewidth=0.5)  # Changed grid style
plt.show()


plt.figure(figsize=(12, 7))
plt.plot(train_acc, color="green", label="Training Accuracy")  # Changed color to green
plt.title("Training Accuracy Over Epochs", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", linewidth=0.5)  # Changed grid style
plt.show()
