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


# pandas 라이브러리 임포트
import pandas as pd


# In[3]:


# Numpy 라이브러리 임포트
import numpy as np


# In[4]:


# matplotlib 라이브러리 임포트
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[6]:


dat = pd.read_csv("./data/ta_20200927155739.csv", skiprows=[0, 1, 2, 3, 4, 5], encoding="cp949")
dat


# In[7]:


# 평균 기온값 추출 및 시각화
temp = dat["평균기온(℃)"]

temp.plot()
plt.show()


# In[8]:


# 데이터 집합을 훈련 데이터와 테스트 데이터로 분할
train_x = temp[:1461] # 20110101 ~ 20141231
test_x = temp[1461:] # 20150101 ~ 20161231


# In[9]:


# Numpy 배열로 변환
train_x = np.array(train_x)
test_x = np.array(test_x)


# In[10]:


# 설명변수의 수
ATTR_SIZE = 180 # 6개월

tmp = []
train_X = []
# 데이터 점 1개 단위로 윈도우를 슬라이드시키며 훈련 데이터를 추출
for i in range(0, len(train_x) - ATTR_SIZE):
    tmp.append(train_x[i:i+ATTR_SIZE])
train_X = np.array(tmp)


# In[11]:


# 훈련 데이터를 데이터프레임으로 변환해서 화면에 출력
pd.DataFrame(train_X)


# In[12]:


# 신경망 구성
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(180, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 180)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# In[13]:


# 인스턴트 생성
model = Net()


# In[14]:


# 오차함수
criterion = nn.MSELoss()


# In[15]:


# 최적화 기볍 선택
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[16]:


# 학습
for epoch in range(1000):
    total_loss = 0
    d = []
    # 훈련 데이터를 미니배치로 분할
    for i in range(100):
        # 훈련 데이터에 인덱스 부여
        index = np.random.randint(0, 1281)
        # 미니배치 분할
        d.append(train_X[index])
    # Numpy 배열로 변환
    d = np.array(d, dtype="float32")
    # 계산 그래프 구성
    d = Variable(torch.from_numpy(d))
    
    # 경사 초기화
    optimizer.zero_grad()
    # 순전파 계산
    output = model(d)
    # 오차 계산
    loss = criterion(output, d)
    # 역전파 계싼
    loss.backward()
    # 가중치 업데이트
    optimizer.step()
    # 오차 누적 계산
    total_loss += float(loss)
    # 100 에포크마다 누적 오차를 출력
    if (epoch + 1) % 100 == 0:
        print(epoch + 1, total_loss)


# In[17]:


# 입력 데이터 플로팅
plt.plot(d.data[0].numpy(), label="original")
plt.plot(output.data[0].numpy(), label="output")
plt.legend(loc="upper right")
plt.show()


# In[18]:


# 이상 점수 계산
tmp = []
test_X = []

# 테스트 데이터를 6개월 단위로 분할
tmp.append(test_x[0:180])
tmp.append(test_x[180:360])
tmp.append(test_x[360:540])
tmp.append(test_x[540:720])
test_X = np.array(tmp, dtype="float32")


# In[19]:


# 데이터를 데이터프레임으로 변환해서 화면에 출력
pd.DataFrame(test_X)


# In[20]:


# 모형 적용
d = Variable(torch.from_numpy(test_X))
output = model(d)


# In[21]:


# 입력 데이터 플로팅
plt.plot(test_X.flatten(), label="original")
plt.plot(output.data.numpy().flatten(), label="prediction")
plt.legend(loc="upper right")
plt.show()


# In[22]:


# 이상 점수 계산
test = test_X.flatten()
pred = output.data.numpy().flatten()

total_score = []
for i in range(0, 720):
    dist = (test[i] - pred[i])
    score = pow(dist, 2)
    total_score.append(score)


# In[23]:


# 이상 점수를 [0, 1] 구간으로 정규화
total_score = np.array(total_score)
max_score = np.max(total_score)
total_score = total_score / max_score


# In[24]:


# 이상 점수 출력
# total_score


# In[25]:


# 이상 점수 플로팅
plt.plot(total_score)
plt.show()

