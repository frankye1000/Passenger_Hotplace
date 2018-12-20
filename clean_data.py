import pandas as pd
import pymssql
import pickle
import numpy as np
train_hire_stats  = pd.read_csv("train_hire_stats.csv")

print(train_hire_stats.head(10))

totalquantity = len(train_hire_stats.values)
year = int(totalquantity/25) #除以25個街區，就是一個街區的一年的小時數(8784小時)


day = int(year/24) #除以24小時，就是一個街區一年天數(366天)

data=[]
for b in range(25):
    blockdata = train_hire_stats.values[ year*b : year*(b+1) ]
    temp=[]
    for i in range(day):
        temp.append(blockdata[24*i:24*(i+1)])
    data.append(np.array(temp))
data = np.array(data)

with open('data.pickle','wb') as fp:
    pickle.dump(data,fp)

