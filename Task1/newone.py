import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import random

# seed 설정
seed = 54 #1 42b 54 82.26
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

BatchSize  = 20
learning_rate = 0.0002 #0.0002
num_epoch = 10



df_train = pd.read_csv('C:/Users/jongh/BCML/DeepLearning/Task1/dataset/fashion-mnist_train.csv', header = 0, index_col = 0)
df_test = pd.read_csv('C:/Users/jongh/BCML/DeepLearning/Task1/dataset/fashion-mnist_test.csv', header = 0, index_col = 0)
x_train = df_train.drop('label', axis = 1).values.reshape(-1, 28, 28)
# labels는 제거할 열의 이름 axis = 0:행 axis = 1:열 / values는 numpy 배열로 전환
y_train = df_train['label'].values

x_test = df_test.drop('label', axis = 1).values.reshape(-1, 28, 28)
y_test = df_test['label'].values

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class NoisyDataset(Dataset):
    def __init__(self, data, labels, noise_factor=0.1, repeats=10):
        self.data = data
        self.labels = labels
        self.noise_factor = noise_factor
        self.repeats = repeats  # 데이터를 몇 배로 늘릴지

    def __len__(self):
        return len(self.data) * self.repeats

    def __getitem__(self, idx):
        original_idx = idx % len(self.data)  # 원본 데이터 인덱스
        img = self.data[original_idx]
        label = self.labels[original_idx]

        # 이미지에 노이즈 추가
        noisy_img = img + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=img.shape)
        noisy_img = np.clip(noisy_img, 0., 1.)  # 값의 범위를 [0, 1]로 유지

        # Tensor로 변환
        noisy_img_tensor = torch.tensor(noisy_img, dtype=torch.float).unsqueeze(0)  # 채널 차원 추가
        label_tensor = torch.tensor(label, dtype=torch.long)

        return noisy_img_tensor, label_tensor


# 원본 데이터를 사용하여 사용자 정의 데이터셋 인스턴스 생성
train_dataset = NoisyDataset(x_train, y_train, noise_factor=0.05, repeats=10)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 배치 정규화 추가
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 배치 정규화 추가
        self.drpout1 = nn.Dropout(p=0.3) #0.3b
        self.drpout2 = nn.Dropout(p=0.5)  # 0.5b 82.98
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=600)
        self.bn3 = nn.BatchNorm1d(600)  # 배치 정규화 추가
        self.fc2 = nn.Linear(in_features=600, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 배치 정규화 적용
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 배치 정규화 적용
        x = x.view(-1, 64 * 7 * 7)
        x = self.drpout1(x)
        x = F.relu(self.bn3(self.fc1(x)))  # 배치 정규화 적용
        x = self.drpout2(x)
        x = self.fc2(x)
        return x


model = CNN_Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)

x_train_tensor = torch.tensor(x_train, dtype=torch.float).unsqueeze(1)  # Add channel dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
#train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
#train_loader = DataLoader(train_dataset, batch_size= BatchSize, shuffle=True)

x_test_tensor = torch.tensor(x_test, dtype=torch.float).unsqueeze(1)  # Add channel dimension
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size= BatchSize, shuffle=False)


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

    print(f'Epoch {epoch+1}/{num_epoch}, Loss: {total_loss/len(train_loader)}')

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