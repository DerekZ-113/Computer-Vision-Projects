import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Define your data directories
train_dir = './data0/train'
test_dir = './data0/test'

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Ensure all images are 64x64
    transforms.RandomHorizontalFlip(),  # Added random horizontal flip for data augmentation to increase variety
    transforms.RandomRotation(10),  # Added random rotation for data augmentation to increase robustness
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images for 3 channels
])

batch_size = 32  # Increased the batch size to stabilize training and make gradient updates smoother

# Load your training data
trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

# Load your testing data
testset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

# Get class names from the dataset
classes = trainset.classes
print('Classes:', classes)

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))
# Print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Increased the number of filters to improve feature extraction capability
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)  # Adjust input features
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, len(classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First convolution and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Second convolution and pooling
        x = self.pool(F.relu(self.conv3(x)))  # Third convolution and pooling
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))   # First fully connected layer
        x = F.relu(self.fc2(x))   # Second fully connected layer
        x = self.fc3(x)           # Output layer
        return x

net = Net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)  # Decreased learning rate for finer updates and to avoid overshooting minima

# Training loop
num_epochs = 20  # Increased the number of epochs to give the model more time to learn
for epoch in range(num_epochs):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data  # Get inputs and labels

        optimizer.zero_grad()  # Zero the parameter gradients

        outputs = net(inputs)       # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()             # Backward pass
        optimizer.step()            # Optimize

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save the trained model
PATH = './model.pth'
torch.save(net.state_dict(), PATH)

# Testing the network on test data
dataiter = iter(testloader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth:', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# Evaluate the network
net.eval()  # Set the network to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # No need to track gradients for evaluation
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # Get predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')

# Calculate accuracy for each class
class_correct = [0. for _ in range(len(classes))]
class_total = [0. for _ in range(len(classes))]
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(len(classes)):
    if class_total[i] > 0:
        print(f'Accuracy of {classes[i]:5s} : {100 * class_correct[i] / class_total[i]:.2f}%')
    else:
        print(f'Accuracy of {classes[i]:5s} : N/A (no samples)')
