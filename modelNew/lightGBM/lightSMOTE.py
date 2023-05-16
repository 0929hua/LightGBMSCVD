import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from lightgbm.sklearn import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from joblib import dump, load
import numpy
def acc(y_test,test_predict):
    Accuracy = accuracy_score(y_test, test_predict)
    print(Accuracy)

    precision=precision_score(y_test, test_predict, average='micro')
    print(precision)

    recall=recall_score(y_test, test_predict, average='micro')
    print(recall)

    f1_micro=f1_score(y_test, test_predict, average='micro')
    print(f1_micro)

    f1_macro=f1_score(y_test, test_predict, average='macro')
    print(f1_macro)

def train(X_train, X_test, y_train, y_test):
    clf_multilabel = OneVsRestClassifier(LGBMClassifier(n_estimators=1000))

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
    dump(model, 'lightGBMSmote.joblib')

#读入数据
dataset = pd.read_csv('../../csvNew/smote/TrainSET.csv', engine='python')

# split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.2, random_state=123)
#
# #模型训练
# train(X_train, X_test, y_train, y_test)


# 模型预测
data = numpy.loadtxt(open("../../csvNew/smote/PredictSET.CSV","rb"),delimiter=",",skiprows=0)

label=numpy.loadtxt(open("../../csvNew/smote/PredictSETLabel.csv","rb"),delimiter=",",skiprows=0)

x_pred = np.array(data)
y_true=np.array(label)

#加载模型
clf = load('lightGBMSmote.joblib')
y_pred = [round(value) for value in clf.predict(x_pred)]
print('y_pred：', y_pred)
acc(y_true, y_pred)
#
#1  # 第一类漏洞数量：2024 第二类漏洞数量：5  第三类漏洞数量：12  第四类漏洞数量：29 第五类漏洞：0  第六类漏洞 ：3
label1=y_pred[0:2073]
print(label1)
m1=0
for i in label1:
    if i==5:
        m1=m1+1
print(m1)

#2  # 第一类漏洞数量：9  第二类漏洞数量：1960  第三类漏洞数量：1  第四类漏洞数量：63 第五类漏洞：24  第六类漏洞 ：16
label2=y_pred[2073:4146]
print(label2)
m2=0
for i in label2:
    if i==5:
        m2=m2+1
print(m2)

#3  # 第一类漏洞数量：30  第二类漏洞数量：18  第三类漏洞数量：1971  第四类漏洞数量：19 第五类漏洞：12  第六类漏洞 ：23
label3=y_pred[4146:6219]
print(label3)
m3=0
for i in label3:
    if i==5:
        m3=m3+1
print(m3)

#4  # 第一类漏洞数量：26  第二类漏洞数量：45  第三类漏洞数量：3  第四类漏洞数量：1992 第五类漏洞：3  第六类漏洞 ：4
label4=y_pred[6219:8292]
print(label4)
m4=0
for i in label4:
    if i==5:
        m4=m4+1
print(m4)

#5  # 第一类漏洞数量：3  第二类漏洞数量：25  第三类漏洞数量：2  第四类漏洞数量：0 第五类漏洞：2043  第六类漏洞 ：0
label5=y_pred[8292:10365]
print(label5)
m5=0
for i in label5:
    if i==5:
        m5=m5+1
print(m5)

#6  # 第一类漏洞数量：5  第二类漏洞数量：8  第三类漏洞数量：2  第四类漏洞数量：8 第五类漏洞：1  第六类漏洞：2049
label6=y_pred[10365:12438]
print(label6)
m6=0
for i in label6:
    if i==5:
        m6=m6+1
print(m6)
Tp1=2024
Tp2=1960
Tp3=1971
Tp4=1992
Tp5=2043
Tp6=2049

Fn1=49
Fn2=113
Fn3=102
Fn4=81
Fn5=30
Fn6=24

Fp1=73
Fp2=101
Fp3=20
Fp4=119
Fp5=40
Fp6=46

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