import pandas as pd
#读入数据
dataset = pd.read_csv('D:/pyProjects/pythonProject/second/csv/smoteNN/allSET.csv', engine='python')
y=dataset.iloc[:, -1]
print(y[0])
m=0
# 0:CALLSTACK  1：可重 2：算数上下溢 3：时间戳 4： TOD 5:TX.origin
#origin 0:1197 1:2084  2:10365 3:3734   4:3992  5:1006
#smote 0:10365 1:10365 2: 10365 3:10365 4:10365 5:10365
#smoteNN 0:9856 1:9556 2: 9643  3:9472  4:10066 5:10082
#smoteTT 0:10305 1:10290 2:10309 3:10295 4:10343 5:10321
for i in y:
    if i==2:
       m=m+1
print(m)
