# from pandas import Series
# from matplotlib import pyplot
# series = Series.from_csv('international-airline-passengers.csv', header=0)
# series.plot()
# pyplot.show()


# from pandas import Series
# from matplotlib import pyplot
# from statsmodels.tsa.seasonal import seasonal_decompose
# series = Series.from_csv('international-airline-passengers.csv', header=0)
# result = seasonal_decompose(series, model='multiplicative')
# result.plot()
# pyplot.show()

import numpy as np
# data_filtered = [[np.array(
#         [[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]])]]
# data_filtered = [[np.array(
#       [[0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.]])]]
# print(np.array(data_filtered[0]))

# a = np.array([1,28,56,3,6,5,4])
# print(np.argmax(a))

import pandas as pd
rain_data = pd.read_csv("rain_data.csv")
print(rain_data)
temp = []
for i in rain_data.values[:,0]:
    temp.extend([i] * 24)
print(len(temp))

pd.DataFrame(temp).to_csv("rain_data_test.csv", header=False, index=False)
