import matplotlib.pyplot as plt



def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % float(height))




if __name__ == '__main__':

    # # LightGBM
    # l1 = [0.9, 0.9679, 0.9728, 0.9907]
    # l2 = [0.899, 0.9679, 0.9728, 0.9907]
    #
    # #RF
    # l1 = [0.8911, 0.9667, 0.9703, 0.9883]
    # l2 = [0.8904, 0.9667, 0.9704, 0.9883]

    # #XGBoost
    # l1 = [0.8950, 0.9575, 0.9634, 0.9866]
    # l2 = [0.8944, 0.9575, 0.9634, 0.9866]

    # #AdaBoost
    # l1 = [0.8039, 0.8566, 0.8750, 0.8973]
    # l2 = [0.7991, 0.8564, 0.8750, 0.8973]

    #SVM
    l1 = [0.3634, 0.5764, 0.5678, 0.6101]
    l2 = [0.2580, 0.5558, 0.5354, 0.6226]

    name = ['Origin SET','SMOTE SET','SMOTETomek SET','SMOTENN SET']
    # total_width, n = 1, 2
    # width = total_width / n
    width = 0.3
    x = [0, 1, 2, 3]
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.rc('font', family='SimHei', size=9)  # 设置中文显示，否则出现乱码！
    a = plt.bar(x, l1, width=width,color = 'steelblue', alpha = 0.4)
    for i in range(len(x)):
        x[i] = x[i] + width+0.1
    b = plt.bar(x, l2, width=width,tick_label=name,color = 'orange', alpha = 0.3)
    autolabel(a)
    autolabel(b)
    plt.xlabel('')
    plt.ylim([0, 1])
    plt.tick_params(axis='x', width=0)
    plt.ylabel('Detection Rate')
    plt.title('')
    # 显示图例
    num1 = 1.05
    num2 = 0
    num3 = 3
    num4 = 0
    plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    plt.show()

