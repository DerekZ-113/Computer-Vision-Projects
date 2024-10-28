import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load the dataset and process it
def load_dataset(data_path):
    images = []
    labels = []

    # Iterate over all class folders
    for class_folder in os.listdir(data_path):
        class_path = os.path.join(data_path, class_folder)
        for img_path in glob.glob(class_path + "/*.jpeg"):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))  # Resize to 64x64
            images.append(img)
            labels.append(class_folder)

    images = np.array(images) / 255.0  # Normalize pixel values
    labels = LabelEncoder().fit_transform(labels)  # Encode labels

    # Split dataset into train and test sets
    return train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 2: Load the dataset
trainX, testX, trainY, testY = load_dataset("./dataset")

# Step 3: Define the CNN model for 4 classes
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):  # Adjusted for 4 classes
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, num_classes)  # Adjust output

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        return torch.softmax(self.fc1(x), dim=1)

# Step 4: Create DataLoader
train_dataset = TensorDataset(
    torch.tensor(trainX).permute(0, 3, 1, 2).float(),  # Shape (N, C, H, W)
    torch.tensor(trainY).long()  # Use long for CrossEntropyLoss
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Step 5: Initialize the model, optimizer, and loss function
model = SimpleCNN(num_classes=4)  # Ensure num_classes=4
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification

# Step 6: Train the model
def train_model(model, train_loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

train_model(model, train_loader, optimizer, criterion)

# Step 7: Save the trained model
torch.save(model.state_dict(), "model.pth")

# Step 8: Load the model for classification
model.load_state_dict(torch.load("model.pth"))
model.eval()
