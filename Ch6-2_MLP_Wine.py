#!/usr/bin/env python
# coding: utf-8

# In[1]:


# PyTorch 라이브러리 임포트
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# In[2]:


# Siikit - learn 라이브러리 임포트
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


# In[3]:


# Pandas 라이브러리 임포트
import pandas as pd


# In[4]:


# 와인 데이터 읽어들이기
wine = load_wine()


# In[5]:


wine_data = wine.data[0:130]
wine_target = wine.target[0:130]


# In[6]:


# 데이터 집합을 훈련 데이터와 테스트 데이터로 분할
train_X, test_X, train_Y, test_Y = train_test_split(wine_data, wine_target, test_size = 0.2)


# In[7]:


# 훈련 데이터 텐서 변환
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()


# In[8]:


# 테스트 데이터 텐서 변환
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()


# In[9]:


# 설명변수와 목적변수의 텐서를 합침
train = TensorDataset(train_X, train_Y)


# In[10]:


# 미니배치로 분할
train_loader = DataLoader(train, batch_size=16, shuffle=True)


# In[11]:


# 신경망 구성
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 96)
        self.fc2 = nn.Linear(96, 96)
        self.fc3 = nn.Linear(96, 96)
        self.fc4 = nn.Linear(96, 96)
        self.fc5 = nn.Linear(96, 96)
        self.fc6 = nn.Linear(96, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x)


# In[12]:


# 인스턴트 생성
model = Net()

# 오차함수 객체
criterion = nn.CrossEntropyLoss()

# 최적화를 담당할 객체
optimizer = optim.SGD(model.parameters(), lr=0.01)


# In[13]:


# 학습 시작
for epoch in range(300):
    total_loss = 0
    # 분할해 둔 데이터를 꺼내옴
    for train_x, train_y in train_loader:
        # 계산 그래프 구성
        # train_x, train_y = Variable(train_x), Variable(train_y)
        # 경사 초기화
        optimizer.zero_grad()
        # 순전파 계산
        output = model(train_x)
        # 오차 계산
        loss = criterion(output, train_y)
        # 역전파 계산
        loss.backward()
        # 가중치 업데이트
        optimizer.step()
        # 누적 오차 계산
        total_loss += float(loss)
        
    # 50회 반복마다 누적 오차 출력:
    if (epoch + 1) % 50 == 0:
        print(epoch + 1, total_loss)


# In[14]:


# 계산 그래프 구성
# test_x, test_y = Variable(test_X), Variable(test_Y)
test_x, test_y = test_X, test_Y


# In[15]:


# 출력이 0 혹은 1이 되게 함
result = torch.max(model(test_x).data, 1)[1]


# In[16]:


# 모형의 정확도 측정
accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())


# In[17]:


# 모형의 정확도 출력
accuracy

