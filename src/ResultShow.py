import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import norm
from torch.autograd import Variable
import scienceplots
from requirements import *
from pearson_calculate import *


def GetResInfo():
    path = "D:/Cloud/Onedrive/OneDrive - emails.bjut.edu.cn/桌面/result/"
    files = os.listdir(path)
    acc = []
    sen = []
    spe = []
    f1 = []
    for file in files:
        if 'Fold' in file:
            data = pd.read_csv(path + file)
            index = list(data['测试集准确率']).index(max(data['测试集准确率']))
            acc.append(list(data['测试集准确率'])[index])
            sen.append(list(data['灵敏度'])[index])
            spe.append(list(data['特异性'])[index])
            f1.append(list(data['F1分数'])[index])
    print("平均准确率:", np.mean(acc))
    print("平均SEN:", np.mean(sen))
    print("平均SPE:", np.mean(spe))
    print("平均F1:", np.mean(f1))


def one_site_test():
    institutions = ['KKI', 'LEUVEN', 'NYU', 'STANFORD', 'Trinity', 'UM', 'USM', 'Yale']
    path = r'D:\Cloud\Onedrive\OneDrive - emails.bjut.edu.cn\桌面\论文中的结果\留一站点验证'
    files = os.listdir(path)
    for institution in institutions:
        for file in files:
            if institution in file:
                print(institution)
                data = pd.read_csv(path + '/' + file)
                acc = eval(data['测试集准确率'].values[0])
                sen = eval(data['灵敏度'].values[0])
                spe = eval(data['特异性'].values[0])
                f1 = eval(data['F1分数'].values[0])
                auc = eval(data['AUC'].values[0])
                index = acc.index(max(acc))
                print("准确率:", max(acc))
                print("sen:", sen[index])
                print("spe:", spe[index])
                print("f1:", f1[index])
                print("auc:", max(auc))
                print("")


def draw():
    # coding=utf-8
    # @Time : 2023/1/31 9:58 PM
    # @Author : 王思哲
    # @File : 柱状图.py
    # @Software: PyCharm

    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 前几个高亮
    THRESHOLD = 0

    # 原始data
    data = \
        pd.read_csv(r'D:\Cloud\Onedrive\OneDrive - emails.bjut.edu.cn\桌面\important_features_sort.csv',
                    encoding='utf-8')[
            "count"].values.tolist()
    # data个数
    data_length = [_ for _ in range(len(data))]
    # 画图数据，根据THRESHOLD分隔
    x1, y1 = data_length[:THRESHOLD], data[:THRESHOLD]
    x2, y2 = data_length[THRESHOLD:], data[THRESHOLD:]

    # 绘图参数
    # plt.style.use('ggplot')  # 添加网格线
    # plt.title("")  # 柱状图标题
    with plt.style.context(["ieee"]):
        plt.xlabel("functional connections")  # X轴名称
        plt.ylabel("count")  # Y轴名称

        print(f"---length--- {len(data)}")
        # plt.bar(x1, y1, width=1, color="#f89515")
        plt.bar(x2, y2, width=1, color="#3482ca")

        plt.savefig('柱状图.png', dpi=500, bbox_inches='tight')
        plt.show()


def unknown():
    path = r'D:\study\大三下\raw_data\rois_aal_csv\Caltech_0051456_rois_aal.csv'
    axis_path = '../description/important_roi.csv'
    data = list(pd.read_csv(path, header=None).iloc[0, :])
    axis = pd.read_csv(axis_path)
    for i in range(len(axis)):
        pos = axis.iloc[i, 0][1:-1].split(',')
        print(data[int(pos[0])], data[int(pos[1])], axis.iloc[i, 1])


def DrawBrain():
    path = "../description/important_roi.csv"
    data = pd.read_csv(path)
    pos = list(data['坐标'])
    pos_x = []
    pos_y = []
    print(pos)
    for i in range(len(pos[:20])):
        x, y = pos[i][1:-1].split(',')
        pos_x.append(int(x))
        pos_y.append(int(y))

    result = np.zeros((116, 116), dtype=int)
    for i in range(len(pos_x)):
        result[pos_x[i]][pos_y[i]] = 1
        result[pos_y[i]][pos_x[i]] = 1
    print(np.array(result.sum()))
    for i in range(116):
        for j in range(116):
            print(f"{result[i][j]}\t", end='')
        print("")


def draw_result():
    # print(plt.style.available)
    # plt.style.use('ieee')
    x = [100, 512, 1024, 1600, 6670]
    # y1 = [0.7090, 0.7448, 0.7622, 0.7463, 0.6716]
    # y2 = [0.6453, 0.7112, 0.6554, 0.6253, 0.4482]
    # y3 = [0.7651, 0.7759, 0.8458, 0.8426, 0.8583]
    # y4 = [0.6814, 0.7049, 0.7176, 0.6957, 0.6052]
    # y5 = [0.7664, 0.8080, 0.8132, 0.7930, 0.7034]

    y1 = [0.7090, 0.6453, 0.7651, 0.7664]
    y2 = [0.7448, 0.7112, 0.7759, 0.8080]
    y3 = [0.7622, 0.6554, 0.8458, 0.8132]
    y4 = [0.7463, 0.6253, 0.8490, 0.7930]
    y5 = [0.6716, 0.4482, 0.8583, 0.7034]

    bar_width = [0.3, 0.3, 0.3, 0.3]
    x1 = [1.0, 3.0, 5.0, 7.0]
    x2 = [1.3, 3.3, 5.3, 7.3]
    x3 = [1.6, 3.6, 5.6, 7.6]
    x4 = [1.9, 3.9, 5.9, 7.9]
    x5 = [2.2, 4.2, 6.2, 8.2]

    with plt.style.context(["ieee"]):
        plt.bar(x1, y1, bar_width, align="center", label="100", alpha=0.5)
        plt.bar(x2, y2, bar_width, align="center", label="512", alpha=0.5)
        plt.bar(x3, y3, bar_width, align="center", label="1024", alpha=0.5)
        plt.bar(x4, y4, bar_width, align="center", label="1600", alpha=0.5)
        plt.bar(x5, y5, bar_width, align="center", label="6700", alpha=0.5)

        plt.xticks(x3, ['ACC', 'SEN', 'SPE', 'AUC'])
        plt.ylim(0.4, 0.9)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
        print("")
        plt.savefig('柱状图.png', dpi=500)
        plt.show()


if __name__ == '__main__':
    DrawBrain()
