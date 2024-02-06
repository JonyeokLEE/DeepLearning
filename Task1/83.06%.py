import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import random

# seed 설정
seed = 54  # 1 42b 54 82.26
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

BatchSize = 20
learning_rate = 0.0002  # 0.0002
num_epoch = 10

df_train = pd.read_csv('C:/Users/jongh/BCML/DeepLearning/Task1/dataset/fashion-mnist_train.csv', header=0, index_col=0)
df_test = pd.read_csv('C:/Users/jongh/BCML/DeepLearning/Task1/dataset/fashion-mnist_test.csv', header=0, index_col=0)
x_train = df_train.drop('label', axis=1).values.reshape(-1, 28, 28)
# labels는 제거할 열의 이름 axis = 0:행 axis = 1:열 / values는 numpy 배열로 전환
y_train = df_train['label'].values

x_test = df_test.drop('label', axis=1).values.reshape(-1, 28, 28)
y_test = df_test['label'].values


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 배치 정규화 추가
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 배치 정규화 추가
        self.drpout1 = nn.Dropout(p=0.3)  # 0.3b
        self.drpout2 = nn.Dropout(p=0.5)  # 0.5b 82.98
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=600)
        self.bn3 = nn.BatchNorm1d(600)  # 배치 정규화 추가
        self.fc2 = nn.Linear(in_features=600, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 배치 정규화 적용
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 배치 정규화 적용
        x = x.view(-1, 64 * 7 * 7)
        x = self.drpout1(x)
        x = F.relu(self.bn3(self.fc1(x)))  # 배치 정규화 적용
        x = self.drpout2(x)
        x = self.fc2(x)
        return x


model = CNN_Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

x_train_tensor = torch.tensor(x_train, dtype=torch.float).unsqueeze(1)  # Add channel dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)

x_test_tensor = torch.tensor(x_test, dtype=torch.float).unsqueeze(1)  # Add channel dimension
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BatchSize, shuffle=False)

for epoch in range(num_epoch):
    model.train()  # 모델을 학습 모드로 설정
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()  # 그라디언트 초기화
        outputs = model(images)  # 모델의 순전파
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 매개변수 업데이트

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epoch}, Loss: {total_loss / len(train_loader)}')

id_list = []
predictions_list = []

model.eval()  # 모델을 평가 모드로 설정
total_correct = 0
total_samples = 0
with torch.no_grad():  # Disable gradient computation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Extend the lists with the batch results
        id_list.extend(range(total_samples, total_samples + labels.size(0)))
        predictions_list.extend(predicted.cpu().numpy())

        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

# Create a DataFrame
results_df = pd.DataFrame({
    'ID': id_list,
    'Label': predictions_list
})

# Save to CSV
results_df.to_csv('JongHyeokLEE.csv', index=False)

total_accuracy = total_correct / total_samples * 100
print(f'Total Accuracy on test set: {total_accuracy:.2f}%')

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