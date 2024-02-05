import numpy as np
import pandas as pd
import tensorflow as tf
import os
for dirname, _, filenames in os.walk('C:/Users/jongh/BCML/DeepLearning/Task1/dataset'):
    for filename in filenames:
        os.path.join(dirname, filename)

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init 

# import matplotlib.pyplot as plt

#set Seed
import random
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

BatchSize  = 20
learning_rate = 0.0002 #0.0002
num_epoch = 10

df_train = pd.read_csv('C:/Users/jongh/BCML/DeepLearning/Task1/dataset/fashion-mnist_train.csv', header = 0, index_col = 0)
df_test = pd.read_csv('C:/Users/jongh/BCML/DeepLearning/Task1/dataset/fashion-mnist_test.csv', header = 0, index_col = 0)

x_train = df_train.drop('label', axis = 1).values.reshape(-1, 28, 28)
y_train = df_train['label'].values

x_test = df_test.drop('label', axis = 1).values.reshape(-1, 28, 28)
y_test = df_test['label'].values

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 600)
        self.fc2 = nn.Linear(600, 10)  # 10 classes in Fashion-MNIST

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN_Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

x_train_tensor = torch.tensor(x_train, dtype=torch.float).unsqueeze(1)  # Add channel dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size= BatchSize, shuffle=True)

x_test_tensor = torch.tensor(x_test, dtype=torch.float).unsqueeze(1)  # Add channel dimension
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size= BatchSize, shuffle=False)

for epoch in range(num_epoch):  # Note: it should be num_epoch, not num_epochs
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{num_epoch}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')