"""Pytorch Analysis"""
# Import all modules you will use
import torchvision # pip install
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch


# create a tensorbaord writer
writer = SummaryWriter()



# Load in the dataset

pick = datasets.MNIST(root='./data', train=True, download=True)

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())#This specifies that we want the test part of the MNIST dataset.
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


image, label = pick[50]

plt.imshow(image, cmap="gray")
plt.title(f"{label}")
plt.show()


train_dataset[1]


# Data cleaning or for NN - work on transforming the data
 #Takes list of Transformations & composes them into single trans(Applied in order). Converts images into tensors. Then normalizes tensor images.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])




# Create Data Loaders on the data to make it easy to feed data to the model in batches.


#make the data loaders

# Loads data in shuffled batches of 64 samples. Reduces overfitting.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)



# init the nn class with an init function and a forward pass
class Net(nn.Module):                      # Defines a new class Net that inherits from nn.Module
    """Define a class to be the NN model (must have an __init__ and a forward function)"""
    def __init__(self):
        """Defines a new class Net that inherits from nn.Module"""
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)   # Input: 28x28 image, Hidden layer: 512 neurons
        self.fc2 = nn.Linear(512, 10)     # Output: 10 classes (digits 0-9)

    def forward(self, x):
        """# specifies how the input data flows through the network layers."""
        x = x.view(-1, 28*28)             # Flatten the image (Removes some of the dimensions)
        x = F.relu(self.fc1(x))         # May not always be the same. ReLU is reliable.
        x = self.fc2(x)                #   Passes the output from the previous layer to the second fully connected layer fc2
        return x                      #   Returns the final output of the network.


model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()


# Initilize the model
model = Net()



# Set the loss function
criterion = nn.CrossEntropyLoss() # BCEWithLogitsLoss could be used as well.



# Set the optimizer function (Brett used "Adam") # could use SGD as well.
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_losses = []
train_accuracies = []
for epoch in range(10):
    model.train()  # Set the model to training mode
    epoch_loss = 0
    CORRECT = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = output.argmax(dim=1)
        CORRECT += pred.eq(target).sum().item()
        total += target.size(0)

    epoch_loss /= len(train_loader)
    accuracy = 100. * CORRECT / total
    train_losses.append(epoch_loss)
    train_accuracies.append(accuracy)

    print(f'Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')



# Path to save the model
MODEL_PATH = "mnist_model.pth"

# Save the model's state_dict
torch.save(model.state_dict(), MODEL_PATH)



## Define the training loop then and print the progress for the loop either every epoch or every 5 or 10 epochs.
#for epoch in range(10): # where I left off @ me
#    for batch_idx, (data, target) in enumerate(train_loader):
#        optimizer.zero_grad()
#        output = model(data)
#        loss = criterion(output, target)
#        loss.backward()
#        optimizer.step()
#    #the song says "print out whats happening"
#    print(f'Epoch {epoch}: Loss: {loss.item():.4f}')


# Evaluate the model with the ".eval()" function
#Calulate the avergae loss and accuracy and print out the results
model.eval()  # Set the model to evaluation mode
TEST_LOSS = 0
CORRECT = 0
with torch.no_grad():  # No gradients needed during evaluation
    for data, target in test_loader:
        output = model(data)
        TEST_LOSS += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        CORRECT += pred.eq(target.view_as(pred)).sum().item()

TEST_LOSS /= len(test_loader.dataset)
accuracy = 100. * CORRECT / len(test_loader.dataset)

print(f'Test Loss: {TEST_LOSS:.4f}, Accuracy: {accuracy:.2f}%')


epochs = range(1, 11)  # 1 to 10

plt.figure(figsize=(12, 6))


# Loss Chart
plt.subplot(1, 1, 1)
plt.plot(epochs, train_losses, label='Training Loss', marker='o')
plt.title('Learning Curve - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(axis='y')


# Accuracy Chart
plt.subplot(1, 1, 1)
plt.plot(epochs, train_accuracies, label='Training Accuracy', marker='o')
plt.title('Learning Curve - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(axis='y')

plt.tight_layout()
plt.show()


