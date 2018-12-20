# LSTM for international airline passengers problem with time step regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 要預測的維度
# 28日*24時=672
outputdim = 672
np.random.seed(7)

# load the dataset
dataframe0 = read_csv('train_hire_stats_8.csv', usecols=[3,4,5,6], engine='python')
dataset = dataframe0.values
dataset = dataset.astype('float32')
print("dataset = \n", dataset)
print("dataset長度 = ", len(dataset))

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print("dataset = \n", dataset)

## 這是給預測的y返回正常值
scalerfor_y = MinMaxScaler(feature_range=(0, 1))
scalerfor_y.fit_transform(dataset[:,0].reshape(-1,1))

t = 720 #(一個月)
predict_time = 672 #(2月)

X = []
y = []
for i in range(len(dataset) - t - predict_time + 1):
    X.append(dataset[i:i + t])                        #t=5
    y.append(dataset[i + t:i + t + predict_time, 0])  #3是你要預測多長的時間

X = np.array(X)
y = np.array(y)

# split into train and test sets
train_size = int(len(X) * 0.8)

X_train, X_test = X[0:train_size,:], X[train_size:len(dataset),:]
y_train, y_test = y[0:train_size,:], y[train_size:len(dataset),:]
print("train_X資料量 = ", len(X_train))
print("test_X資料量 = ", len(X_test))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
# model.add(Dropout(0.2))
# model.add(Dense(units=512,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=256,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=predict_time))
print(model.summary())

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1, batch_size=100, verbose=2)

# make predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
# print("trainPredict = ", trainPredict)
# print("y_train = ", y_train)

# print("testPredict = ", testPredict[-1])
# print("y_test = ", y_test[-1])
print("便回正常值_預測的", scalerfor_y.inverse_transform(testPredict[-1].reshape(-1,1)))
print("變回正常值_真的", scalerfor_y.inverse_transform(y_test[-1].reshape(-1,1)))

pd.DataFrame(trainPredict).to_csv("trainPredict.csv", header=False, index=False)
# pd.DataFrame(X_test).to_csv("X_test.csv", header=False, index=False)
pd.DataFrame(testPredict).to_csv("testPredict.csv", header=False, index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", header=False, index=False)


# print("---------------------------------------------")
# print("testY", testY)
# calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
