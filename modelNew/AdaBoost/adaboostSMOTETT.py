import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from joblib import dump, load
import numpy
def acc(y_test,test_predict):
    Accuracy = accuracy_score(y_test, test_predict)
    print("Accuracy :")
    print(Accuracy)

    precision = precision_score(y_test, test_predict, average='micro')
    print("precision :")
    print(precision)

    recall = recall_score(y_test, test_predict, average='micro')
    print("recall :")
    print(recall)

    f1_micro = f1_score(y_test, test_predict, average='micro')
    print("f1_micro :")
    print(f1_micro)

    f1_macro = f1_score(y_test, test_predict, average='macro')
    print("f1_macro :")
    print(f1_macro)

def train(X_train, X_test, y_train, y_test):
    clf_multilabel = OneVsRestClassifier(AdaBoostClassifier(n_estimators=500))

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
    dump(model, 'adaboostSmoteTT.joblib')

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
clf = load('adaboostSmoteTT.joblib')
y_pred = [round(value) for value in clf.predict(x_pred)]
print('y_pred：', y_pred)
acc(y_true, y_pred)

#
#1  # 第一类漏洞数量：1760  第二类漏洞数量：9  第三类漏洞数量：83  第四类漏洞数量：142 第五类漏洞：13  第六类漏洞 ：55
label1=y_pred[0:2062]
print(label1)
m=0
for i in label1:
    if i==5:
        m=m+1
print(m)

#2  # 第一类漏洞数量：36  第二类漏洞数量：1756  第三类漏洞数量：6  第四类漏洞数量：158 第五类漏洞：61  第六类漏洞 ：45
label1=y_pred[2062:4124]
print(label1)
m=0
for i in label1:
    if i==5:
        m=m+1
print(m)

#3  # 第一类漏洞数量：128  第二类漏洞数量：32  第三类漏洞数量：1732 第四类漏洞数量：10 第五类漏洞：9  第六类漏洞 ：151
label1=y_pred[4124:6186]
print(label1)
m=0
for i in label1:
    if i==5:
        m=m+1
print(m)

#4  # 第一类漏洞数量：96  第二类漏洞数量：110  第三类漏洞数量：5  第四类漏洞数量：1840 第五类漏洞：1  第六类漏洞 ：10
label1=y_pred[6186:8248]
print(label1)
m=0
for i in label1:
    if i==5:
        m=m+1
print(m)

#5  # 第一类漏洞数量：13  第二类漏洞数量：47  第三类漏洞数量：8  第四类漏洞数量：0 第五类漏洞：1991  第六类漏洞 ：3
label1=y_pred[8248:10310]
print(label1)
m=0
for i in label1:
    if i==5:
        m=m+1
print(m)

#6  # 第一类漏洞数量：91  第二类漏洞数量：52  第三类漏洞数量：131  第四类漏洞数量：38 第五类漏洞：3  第六类漏洞：1747
label1=y_pred[10310:12372]
print(label1)
m=0
for i in label1:
    if i==5:
        m=m+1
print(m)

#2062
Tp1=1760
Tp2=1756
Tp3=1732
Tp4=1840
Tp5=1991
Tp6=1747
#
Fn1=302
Fn2=306
Fn3=330
Fn4=222
Fn5=71
Fn6=315
#
Fp1=364
Fp2=250
Fp3=233
Fp4=348
Fp5=87
Fp6=264

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
