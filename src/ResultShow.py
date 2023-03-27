import numpy as np
import pandas as pd
import re
from requirements import *
from pearson_calculate import *


def DrawROC(fpr, tpr, best_auc):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="AUC={:.3f}".format(best_auc))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC curve')
    plt.legend()
    plt.savefig(f"test.png", dpi=500)
    plt.show()


def GetResInfo():
    path = r"C:\Users\Chen\Desktop\CC200\CC200_h8/"
    # path = "../result/"
    files = os.listdir(path)
    acc = []
    sen = []
    spe = []
    f1 = []
    auc = []
    best_fpr = None
    best_tpr = None
    best_auc = -1
    for file in files:
        if '2_Fold' in file:
            data = pd.read_csv(path + file)
            index = list(data['测试集准确率']).index(max(data['测试集准确率']))
            acc.append(list(data['测试集准确率'])[index])
            sen.append(list(data['灵敏度'])[index])
            spe.append(list(data['特异性'])[index])
            f1.append(list(data['F1分数'])[index])
            auc.append(list(data['AUC'])[index])

            if best_auc < auc[-1]:
                best_auc = auc[-1]
                best_fpr = data['FPR'][index]
                best_tpr = data['TPR'][index]

    temp = best_fpr[1:-1].split(" ")
    fpr = []
    for item in temp:
        if item == '0.':
            fpr.append(0)
        elif '.' in item:
            fpr.append(float(item))

    temp = best_tpr[1:-1].split(" ")
    tpr = []
    for item in temp:
        if item == '0.':
            tpr.append(0)
        elif '.' in item:
            tpr.append(float(item))

    DrawROC(fpr, tpr, best_auc)

    print("平均准确率:", np.mean(acc))
    print("平均SEN:", np.mean(sen))
    print("平均SPE:", np.mean(spe))
    print("平均F1:", np.mean(f1))
    print("平均AUC", np.mean(auc))
    print(fpr)
    print(tpr)
    print(best_auc)


def one_site_error():
    institutions = ['KKI', 'LEUVEN', 'NYU', 'STANFORD', 'Trinity', 'UM', 'USM', 'Yale']
    path = r'C:\Users\Chen\Desktop\CC200\one_site'
    files = os.listdir(path)
    total = pd.DataFrame(columns=['Institution', 'ACC', 'SEN', 'SPE', 'AUC'])
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')

    for institution in institutions:
        for file in files:
            if institution in file:
                raw_data = pd.read_csv(path + '/' + file)
                names = raw_data.columns[1:]
                FPR = []
                temp = list(raw_data['FPR'])[0][6:].split("array")
                for i, item in enumerate(temp):
                    if i != len(temp) - 1:
                        FPR.append(eval(item[1:-3]))
                    else:
                        FPR.append(eval(item[1:-2]))

                TPR = []
                temp = list(raw_data['TPR'])[0][6:].split("array")
                for i, item in enumerate(temp):
                    if i != len(temp) - 1:
                        TPR.append(eval(item[1:-3]))
                    else:
                        TPR.append(eval(item[1:-2]))

                data = pd.DataFrame()
                for i in range(len(names) - 2):
                    data[names[i]] = eval(raw_data.iloc[0, i + 1])

                index = list(data['测试集准确率']).index(max(data['测试集准确率']))
                total.loc[len(total)] = {'Institution': institution,
                                         'ACC': data['测试集准确率'][index],
                                         'SEN': data['灵敏度'][index],
                                         'SPE': data['特异性'][index],
                                         'AUC': data['AUC'][index]}

                plt.plot(FPR[index], TPR[index], label="{}: {:.3f}".format(institution, data['AUC'][index]),
                         linewidth=1)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC curve')
    plt.legend()
    plt.savefig(f"test.png", dpi=500)
    plt.show()
    print(total)


def one_site_test():
    institutions = ['KKI', 'LEUVEN', 'NYU', 'STANFORD', 'Trinity', 'UM', 'USM', 'Yale']
    path = r'C:\Users\Chen\Desktop\CC200\one_site'
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
                best_fpr = data['FPR'][index]
                best_tpr = data['TPR'][index]
                temp = best_fpr[1:-1].split(" ")
                fpr = []
                for item in temp:
                    if item == '0.':
                        fpr.append(0)
                    elif '.' in item:
                        fpr.append(float(item))

                temp = best_tpr[1:-1].split(" ")
                tpr = []
                for item in temp:
                    if item == '0.':
                        tpr.append(0)
                    elif '.' in item:
                        tpr.append(float(item))
                print("准确率:", max(acc))
                print("sen:", sen[index])
                print("spe:", spe[index])
                print("f1:", f1[index])
                print("auc:", max(auc))
                print("")

                print("FPR", fpr)
                print("TPR", tpr)


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


if __name__ == '__main__':
    GetResInfo()
    # strings = [['a', 'as', 'bat', 'car', 'dove'], ['b', 'bs', 'bbt']]
    # res = [item.upper() for string in strings for item in string if len(item) > 2]
    # print(res)
