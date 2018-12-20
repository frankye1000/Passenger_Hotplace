import pandas as pd
import numpy as np
import datetime

## 創造日期列表
def createdatelist(start = '2016-01-31', end = '2017-01-31'):
    datestart = datetime.datetime.strptime(start, '%Y-%m-%d')
    dateend = datetime.datetime.strptime(end, '%Y-%m-%d')

    datelist = []
    while datestart < dateend:
        datestart += datetime.timedelta(days=1)
        datelist.append(datestart.strftime('%Y-%m-%d'))
    return datelist

datelist = createdatelist()
## 爬取一年每日溫度
def crawltemperature():
    yeartemplist = []
    for date in datelist:
        ## 利用pd爬取溫度資料
        url = "https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=C0A9F0&stname=%25E5%2585%25A7%25E6%25B9%2596&datepicker={}".format(date)
        df = pd.read_html(url)[1][3:].values
        df = df[:,3:4]
        ## 00:00 的資料要移到每天第一筆
        dailytemp_array = np.concatenate((df[-1:], df[:-1]), axis = 0)
        # print(dailytemp_array)
        # print(list(dailytemp_array))
        ## '/','X'資料要補成平均溫度
        dailytemp_list = [float(v[0]) for v in dailytemp_array if v[0] != '/' and v[0] != 'X']
        mean = round(sum(dailytemp_list) / len(dailytemp_list), 1)
        emptyindex = [i for i, v in enumerate(dailytemp_array) if v[0] == '/' or v[0] == 'X']

        dailytemp_list = [float(v[0]) if v[0] != '/' and v[0] != 'X' else mean for v in dailytemp_array ]
        yeartemplist.extend(dailytemp_list)
        print("finish :",date)
        return yeartemplist

# 3/11 3/12 3/13 這三天手動補14度
# 6/1 補26度
if __name__ == '__main__':
    yeartemplist = crawltemperature()
    yeartempdf = pd.DataFrame(yeartemplist)
    yeartempdf.to_csv("temperature.csv", index=False, header=False)



