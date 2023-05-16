# 导入绘图模块
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
# 构建数据
def to_percent(temp, position):
 return '%1.0f'%(10*temp) + '%'
# TPR1:
# 0.9902813299232737
# FPR1:
# 0.001125319693094629

# TPR2:
# 0.9820971867007673
# FPR2:
# 0.0032736572890025577

# TPR3:
# 0.9948849104859335
# FPR3:
# 0.00020460358056265986

# TPR4:
# 0.9892583120204603
# FPR4:
# 0.003989769820971867

# TPR5:
# 0.9938618925831202
# FPR5:
# 0.001227621483375959

# TPR6:
# 0.9948849104859335
# FPR6:
# 0.001125319693094629

Y2016 = [0.99, 0.982,  0.995, 0.989,0.994,0.995]
Y2017 = [0.999, 0.997, 1, 0.996,0.999,0.999]
Y2018 = [0.01, 0.018,0.005, 0.011,0.006,0.005]
Y2019 = [0.001, 0.003,  0.0, 0.004,0.001,0.001]
labels = ['Callstck','Reentrancy','overflow','Timestamp','TOD','TX.origin']
bar_width = 0.3
width = 0.1
# list2 = [94, 99.3, 94.3, 71.3]
# list3 = [6, 0.6,5.7, 28.7]
# list4 = [99.3, 99.6, 100, 100]
# list5 = [0.7, 0.4,  0, 0]
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 中文乱码的处理
plt.rcParams['font.sans-serif'] =[u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 绘图12
plt.bar(np.arange(6), Y2016, label = 'TPR',   color = 'steelblue', alpha = 0.4, width = bar_width)
plt.bar(np.arange(6), Y2018, label = 'FNR',bottom=Y2016,color = 'steelblue',alpha = 0.4,width = bar_width,hatch="xxx")

plt.bar(np.arange(6)+width+bar_width, Y2017, label = 'TNR', color = 'orange', alpha = 0.3, width = bar_width)
plt.bar(np.arange(6)+width+bar_width, Y2019, label = 'FPR',bottom=Y2017 ,color = 'orange', alpha = 0.3, width = bar_width,hatch="xxx")

# 添加轴标签
plt.xlabel('Six Vulnerabilities')
plt.ylabel('Detection Rate')
# 添加刻度标签
plt.xticks(np.arange(6)+bar_width,labels)
# 设置Y轴的刻度范围
plt.ylim([0.95, 1])
#plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
# 显示图例
num1=1.05
num2=0
num3=3
num4=0
plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
# 显示图形
fig.tight_layout()  # 调整整体空白
plt.show()
