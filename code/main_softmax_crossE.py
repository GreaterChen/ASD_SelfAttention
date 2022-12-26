import sys
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from prettytable import PrettyTable
from tqdm import tqdm
from utils import draw_result_pic
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from GetData import GetData
from Module import Module
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

dataset_size = 20  # 训练的样本总数,-1代表全部训练
batch_size = 1  # 每次训练样本数
Head_num = 1  # self-attention的头数
Windows_num = 145  # 时间窗的个数
Vector_len = int(116 * 115 / 2)  # 上三角展开后的长度
data_num = -1  # 数据集个数(自动获取)
epoch = 2  # 训练轮次
learn_rate = 0.01

# root_path = "/root/autodl-tmp/rois_aal_csv_pearson"
# label_path = "label_678.csv"

root_path = "../raw_data/rois_aal_csv_pearson"
label_path = "../description/label_678.csv"


def Entire_main():
    all_data = GetData(root_path, label_path,dataset_size)  # 一次性读取所有数据

    kf = KFold(n_splits=5, shuffle=True, random_state=0)  # 初始化5折交叉验证的工具

    # 对每一折进行记录
    train_acc_list_kf = []
    train_loss_list_kf = []
    test_acc_list_kf = []
    test_loss_list_kf = []

    split_range = 0
    last_time = 0
    for train_index, test_index in kf.split(all_data):  # 此处获取每一折的索引
        # 对于每一折来说，都要从0开始训练模型
        # 因为如果不同折训练同一个模型，会出现当前折的测试集曾被另一折当作训练集训练，导致准确率异常
        module = Module()
        module = module.cuda()

        # 损失函数：交叉熵
        loss_fn = nn.CrossEntropyLoss()
        loss_fn = loss_fn.cuda()
        # 优化器：SGD
        optimizer = torch.optim.SGD(module.parameters(), lr=learn_rate)

        p_table = PrettyTable(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "time(s)"])

        # 此处获取真正的该折数据
        train_fold = Subset(all_data, train_index)
        test_fold = Subset(all_data, test_index)

        train_size = len(train_index)
        test_size = len(test_index)

        train_dataloader = DataLoader(train_fold, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_fold, batch_size=batch_size, shuffle=True)

        split_range += 1

        # 对该折的所有轮进行记录
        train_acc_list = []
        train_loss_list = []
        test_acc_list = []
        test_loss_list = []

        # 下面开始当前折的训练
        for epoch_i in range(epoch):
            # 对该折该轮的所有dataloader进行记录
            epoch_train_loss = 0
            epoch_train_acc = 0
            epoch_test_loss = 0
            epoch_test_acc = 0

            module.train()
            # 下面开始当前折、当前轮的训练，即以batch_size的大小进行训练
            for data in tqdm(train_dataloader, desc=f'Fold{split_range}-Epoch{epoch_i + 1}', file=sys.stdout):
                x, y = data
                x = x.cuda()
                y = y.cuda()
                y = y.to(torch.float32)  # 这一步似乎很费时间

                output = module(x)

                loss = loss_fn(output, y)
                epoch_train_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.step()
                train_acc = 0
                for i, res in enumerate(output):
                    if res[0] > res[1]:
                        train_acc = train_acc + 1 if y[i][0] > y[i][1] else train_acc
                    if res[0] < res[1]:
                        train_acc = train_acc + 1 if y[i][0] < y[i][1] else train_acc
                epoch_train_acc += train_acc

            # 记录当前折、当前轮的数据
            train_acc_list.append(float(epoch_train_acc / train_size))
            train_loss_list.append(float(epoch_train_loss))

            module.eval()  # 设置测试模式
            with torch.no_grad():
                for data in test_dataloader:
                    x, y = data
                    x = x.cuda()
                    y = y.cuda()
                    y = y.to(torch.float32)

                    output = module(x)

                    loss = loss_fn(output, y)
                    epoch_test_loss += loss
                    test_acc = 0
                    for i, res in enumerate(output):
                        if res[0] > res[1]:
                            test_acc = test_acc + 1 if y[i][0] > y[i][1] else test_acc
                        if res[0] < res[1]:
                            test_acc = test_acc + 1 if y[i][0] < y[i][1] else test_acc
                    epoch_test_acc += test_acc
            test_acc_list.append(float(epoch_test_acc / test_size))
            test_loss_list.append(float(epoch_test_loss))

            p_table.add_row(
                [epoch_i + 1, float(train_loss_list[-1]), float(train_acc_list[-1]), float(test_loss_list[-1]),
                 float(test_acc_list[-1]), time.time() - last_time])
            last_time = time.time()
            print(p_table)
            if epoch_i == epoch - 1:
                with open("result.txt", "a") as f:
                    f.write(str(p_table))
                    f.write("\n")

        # 记录每一折的数据，是一个二维的列表
        train_acc_list_kf.append(train_acc_list)
        train_loss_list_kf.append(train_loss_list)
        test_acc_list_kf.append(test_acc_list)
        test_loss_list_kf.append(test_loss_list)

    avg_train_acc = np.array(train_acc_list_kf).mean(axis=0)
    avg_train_loss = np.array(train_loss_list_kf).mean(axis=0)
    avg_test_acc = np.array(test_acc_list_kf).mean(axis=0)
    avg_test_loss = np.array(test_loss_list_kf).mean(axis=0)

    with open("result_avg.txt", "a") as f:
        f.write(str(avg_train_acc))
        f.write(str(avg_train_loss))
        f.write(str(avg_test_acc))
        f.write(str(avg_test_loss))
        f.write('\n')

    # 传结果list格式： [train(list), test(list)]
    draw_result_pic(save_path='image/data166_epoch10_acc.png', res=[avg_train_acc, avg_test_acc], start_epoch=5,
                    pic_title='acc')
    draw_result_pic(save_path='image/data166_epoch10_loss.png', res=[avg_train_loss, avg_test_loss], start_epoch=5,
                    pic_title='loss')


if __name__ == '__main__':
    Entire_main()