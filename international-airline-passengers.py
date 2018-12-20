# LSTM for international airline passengers problem with time step regression framing
import numpy
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

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-outputdim-1):
		a = dataset[i:(i+look_back), 0]
		b = dataset[i+look_back:i+look_back+outputdim, 0] #672=28天*24時

		dataX.append(a)
		dataY.append(b)

	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)

# 創造要預測的dataset_X
# def create_predict_dataset(dataset, look_back=1):
# 	dataX = []
# 	for i in range(len(dataset)-look_back-1):
# 		a = dataset[i:(i+look_back), 0]
# 		dataX.append(a)
# 	return numpy.array(dataX)


# load the dataset
dataframe0 = read_csv('train_hire_stats_8.csv', usecols=[3], engine='python')
dataset = dataframe0.values
dataset = dataset.astype('float32')
print("dataset = \n", dataset)
print("dataset長度 = ", len(dataset))
# # load the predict_dataset
# dataframe1 = read_csv('test_hire_stats.csv', usecols=[1,3], engine='python')
# dataframe1 = dataframe1[dataframe1['Zone_ID'] == 7]['Hour_slot']
# # print(dataframe1)
# predict_dataset = dataframe1.values
# predict_dataset = predict_dataset.astype('float32')
# # print(predict_dataset)


# # 每日溫度與平均溫度差
# mean_temperature = numpy.mean(dataset[:, 1:2])
# temperature = numpy.fabs(dataset[:, 1:2] - mean_temperature)

# normalize the dataset
scaler_quantity = MinMaxScaler(feature_range=(0, 1))
dataset_quantity = scaler_quantity.fit_transform(dataset[:, :1])
# print(dataset_quantity)
# scaler_temperature = MinMaxScaler(feature_range=(0, 1))
# dataset_temperature = scaler_temperature.fit_transform(temperature)
# dataset = numpy.concatenate((dataset_quantity, dataset_temperature), axis = 1)
dataset = dataset_quantity
print("dataset = \n", dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# train = [[0.1],[0.2],[0.3].....]
print("train資料量 = ", len(train))
print("test資料量 = ", len(test))

# reshape into X=t and Y=t+1
look_back = 720 #24時*30日=720
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print("trainX_shape = ", trainX.shape)
print("trainY_shape = ", trainY.shape)
print("trainX資料量 = ", len(trainX))
print("testY資料量 = ", len(testY))

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# print(trainX)
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(1024, input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(Dense(units=outputdim))
print(model.summary())


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=10, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
# trainPredict = scaler_quantity.inverse_transform(trainPredict)
# trainY = scaler_quantity.inverse_transform([trainY])
testPredict = scaler_quantity.inverse_transform(testPredict)
# testY = scaler_quantity.inverse_transform([testY])
# print("trainPredict = ", trainPredict)
# print("---------------------------------------------")
# print("trainY = ", trainY)
# print("---------------------------------------------")
print("testPredict = \n", testPredict[-1])
df_predict = pd.DataFrame(testPredict[-1])
df_predict.to_csv("test_hire_stats_17.csv", header=False, index=False)


# print("---------------------------------------------")
# print("testY", testY)
# calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
#
#
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# # plot baseline and predictions
# plt.plot(scaler_quantity.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()