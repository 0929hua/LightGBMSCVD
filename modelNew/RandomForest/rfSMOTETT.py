import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from joblib import dump, load
import numpy
def acc(y_test,test_predict):
    Accuracy = accuracy_score(y_test, test_predict)
    print(Accuracy)

    precision = precision_score(y_test, test_predict, average='micro')
    print(precision)

    recall = recall_score(y_test, test_predict, average='micro')
    print(recall)

    f1_micro = f1_score(y_test, test_predict, average='micro')
    print(f1_micro)

    f1_macro = f1_score(y_test, test_predict, average='macro')
    print(f1_macro)

def train(X_train, X_test, y_train, y_test):
    clf_multilabel = OneVsRestClassifier(RandomForestClassifier())

    model = clf_multilabel.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(pred)
    # 模型评估
    # error_rate=np.sum(pred!=test.lable)/test.lable.shape[0]
    error_rate = np.sum(pred != y_test) / y_test.shape[0]
    print('测试集错误率(softmax):{}'.format(error_rate))

    accuray = 1 - error_rate
    print('测试集准确率：%.4f' % accuray)
    # 模型保存
    dump(model, 'rfSmoteTT.joblib')

#读入数据
dataset = pd.read_csv('../../csvNew/smoteTT/TrainSET.csv', engine='python')

# split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.2, random_state=123)

# #模型训练
# train(X_train, X_test, y_train, y_test)


# 模型预测
data = numpy.loadtxt(open("../../csvNew/smoteTT/PredictSET.CSV","rb"),delimiter=",",skiprows=0)

label=numpy.loadtxt(open("../../csvNew/smoteTT/PredictSETLabel.csv","rb"),delimiter=",",skiprows=0)

x_pred = np.array(data)
y_true=np.array(label)

#加载模型
clf = load('rfSmoteTT.joblib')
y_pred = [round(value) for value in clf.predict(x_pred)]
print('y_pred：', y_pred)
acc(y_true, y_pred)
#0.9749

#1  # 第一类漏洞数量：2010  第二类漏洞数量：4  第三类漏洞数量：13  第四类漏洞数量：30 第五类漏洞：2  第六类漏洞 ：3
label1=y_pred[0:2062]
print(label1)
m=0
for i in label1:
    if i==5:
        m=m+1
print(m)

#2  # 第一类漏洞数量：9  第二类漏洞数量：1948  第三类漏洞数量：3 第四类漏洞数量：70 第五类漏洞：19  第六类漏洞 ：13
label1=y_pred[2062:4124]
print(label1)
m=0
for i in label1:
    if i==5:
        m=m+1
print(m)

#3  # 第一类漏洞数量：21  第二类漏洞数量：19  第三类漏洞数量：1989 第四类漏洞数量：11 第五类漏洞：6  第六类漏洞 ：16
label1=y_pred[4124:6186]
print(label1)
m=0
for i in label1:
    if i==5:
        m=m+1
print(m)

#4  # 第一类漏洞数量：30  第二类漏洞数量：37  第三类漏洞数量：2  第四类漏洞数量：1991 第五类漏洞：0  第六类漏洞 ：2
label1=y_pred[6186:8248]
print(label1)
m=0
for i in label1:
    if i==5:
        m=m+1
print(m)

#5  # 第一类漏洞数量：3  第二类漏洞数量：15  第三类漏洞数量：6  第四类漏洞数量：0 第五类漏洞：2038  第六类漏洞 ：0
label1=y_pred[8248:10310]
print(label1)
m=0
for i in label1:
    if i==5:
        m=m+1
print(m)

#6  # 第一类漏洞数量：7  第二类漏洞数量：9  第三类漏洞数量：6  第四类漏洞数量：11 第五类漏洞：0  第六类漏洞：2029
label1=y_pred[10310:12372]
print(label1)
m=0
for i in label1:
    if i==5:
        m=m+1
print(m)
#完成

#2062
Tp1=2010
Tp2=1948
Tp3=1989
Tp4=1991
Tp5=2038
Tp6=2029
#
Fn1=52
Fn2=114
Fn3=73
Fn4=71
Fn5=24
Fn6=33
#
Fp1=70
Fp2=84
Fp3=30
Fp4=122
Fp5=27
Fp6=34

tp=Tp1+Tp2+Tp3+Tp4+Tp5+Tp6
fp=Fp1+Fp2+Fp3+Fp4+Fp5+Fp6
fn=Fn1+Fn2+Fn3+Fn4+Fn5+Fn6
sum=2*tp+fp+fn
micro1=2*tp/sum
print("micro1: " + str(micro1))

sum1=2*Tp1+Fp1+Fn1
f1=(2*Tp1)/sum1
print("f1: ")
print(f1)

sum2=2*Tp2+Fp2+Fn2
f2=2*Tp2/sum2
print("f2: ")
print(f2)

sum3=2*Tp3+Fp3+Fn3
f3=2*Tp3/sum3
print("f3: ")
print(f3)

sum4=2*Tp4+Fp4+Fn4
f4=2*Tp4/sum4
print("f4: ")
print(f4)

sum5=2*Tp5+Fp5+Fn5
f5=2*Tp5/sum5
print("f5: ")
print(f5)

sum6=2*Tp6+Fp6+Fn6
f6=2*Tp6/sum6
print("f6: ")
print(f6)


sumF1=f1+f2+f3+f4+f5+f6
macro1=sumF1/6
print("macro1: " +str(macro1))
