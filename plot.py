import matplotlib.pyplot as plt
import pickle
with open('data.pickle','rb') as fp:
    data = pickle.load(fp)
# 10 9 8 3 5 街區人數 占總比例60%
# print(data[2])

## 把資料整理成[[一天載客量], [一天載客量]]
quantity_passenger_1 = [hour[3] for day in data[2][:2] for hour in day]
quantity_passenger_2 = [hour[3] for day in data[4][:2] for hour in day]
quantity_passenger_3 = [hour[3] for day in data[7][:2] for hour in day]
quantity_passenger_4 = [hour[3] for day in data[8][:2] for hour in day]
quantity_passenger_5 = [hour[3] for day in data[9][:2] for hour in day]


plt.figure(figsize=(10, 20))
plt.ylabel("passenger")
plt.xlabel("time")
plt.plot(quantity_passenger_1)
plt.plot(quantity_passenger_2)
plt.plot(quantity_passenger_3)
plt.plot(quantity_passenger_4)
plt.plot(quantity_passenger_5)
plt.show()