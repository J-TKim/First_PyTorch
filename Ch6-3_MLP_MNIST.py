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


# scikit-learn 라이브러리 임포트
from sklearn.datasets import load_digits
from sklearn import datasets, model_selection


# In[3]:


# pandas 라이브러리 임포트
import pandas as pd


# In[4]:


# numpy 라이브러리 임포트
import numpy as np


# In[5]:


# matplotlib 라이브러리 임포트
from matplotlib import pyplot as plt
from matplotlib import cm


# In[6]:


# 이미지를 노트북 안에 출력하도록 함
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[8]:


# MNIST 데이터를 읽어 들인 다음, 화면에 출력
mnist = datasets.fetch_openml('mnist_784', data_home="./data", version=1, cache=True)

#mnist


# In[9]:


# 설명변수를 정규화하고 변수에 대입하고 화면에 출력
mnist_data = mnist.data / 255


# In[10]:


# 데이터프레임 객체로 변환하고 화면에 출력
pd.DataFrame(mnist_data)


# In[11]:


# 1번째 이미지를 화면에 출력
plt.imshow(mnist_data[0].reshape(28, 28), cmap=cm.gray_r)
plt.show()


# In[12]:


# 목적변수를 변수에 할당하고 데이터를 화면에 출력
mnist_label = mnist.target
mnist_label


# In[13]:


# 훈련 데이터 건수
train_size = 5000

# 테스트 데이터 건수
test_size = 500


# In[14]:


# 데이터 집합을 훈련 데이터와 테스트 데이터로 분할
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(
    mnist_data, mnist_label, train_size=train_size,test_size=test_size)


# In[15]:


# 훈련 데이터 텐서 변환
train_X = torch.tensor(train_X, dtype=torch.float) / 255
train_Y = torch.tensor([int(x) for x in train_Y])


# In[16]:


# 테스트 데이터 텐서 변환
test_X = torch.tensor(test_X, dtype=torch.float) / 255
test_Y = torch.tensor([int(x) for x in test_Y])


# In[17]:


train_X = train_X.to(device)
train_Y = train_Y.to(device)
test_X = test_X.to(device)
test_Y = test_Y.to(device)


# In[18]:


# 변환된 텐서의 데이터 건수 확인
print(train_X.shape)
print(train_Y.shape)


# In[19]:


# 설명변수와 목적변수 텐서를 합침
train = TensorDataset(train_X, train_Y)


# In[20]:


# 텐서의 첫 번째 데이터를 확인
# print(train[0])


# In[21]:


# 미니배치 분할
train_loader = DataLoader(train, batch_size=100, shuffle=True)


# In[22]:


# 신경망 구성
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.dropout(x, training=self.training)
        x = self.fc6(x)
        return F.log_softmax(x)


# In[23]:


# 인스턴트 생성
model = Net().to(device)


# In[24]:


# 오차함수 객체
criterion = nn.CrossEntropyLoss()


# In[25]:


# 최적화를 담당할 객체 
optimizer = optim.SGD(model.parameters(), lr=0.01)


# In[26]:


# 학습 시작
for epoch in range(1000):
    total_loss = 0
    # 분할해 둔 데이터를 꺼내옴
    for train_x, train_y in train_loader:
        # 계산 그래프 구성
        train_x, train_y = Variable(train_x),  Variable(train_y)
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
    # 100회 반복마다 누적 오차 출력
    if (epoch + 1) % 100 == 0:
        print(epoch + 1, total_loss)


# In[27]:


# 계산 그래프 구성
test_x, test_y = Variable(test_X), Variable(test_Y).to("cpu")


# In[28]:


# 출력이 0 또는 1이 되게 함
result = torch.max(model(test_x).data, 1)[1]


# In[29]:


# 모형의 정확도 측정
acc = sum(test_y.data.numpy() == result.to("cpu").numpy()) / len(test_y.data.numpy())


# In[30]:


# 모형의 정확도 출력
acc

