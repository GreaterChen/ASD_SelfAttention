import math
import os
import sys

import numpy as np
import pandas as pd
import torchvision
import time
import torch
from torch import nn
import torch.nn.functional as F
import torchviz
from torch.utils.data import Dataset, DataLoader
from prettytable import PrettyTable
from tqdm import tqdm
from utils import draw_result_pic
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset

batch_size = 2  # 每次训练样本数
Head_num = 1  # self-attention的头数
Windows_num = 145  # 时间窗的个数
Vector_len = int(116 * 115 / 2)  # 上三角展开后的长度
data_num = -1  # 数据集个数(自动获取)

root_path = "raw_data/rois_aal_csv_pearson_NYU"
label_path = "description/label.csv"


class GetData(Dataset):
    def __init__(self, root_path, label_path):
        '''
        继承于Dataset,与主函数中DataLoader一起取数据
        :param root_path: 数据集目录地址
        :param label_path: 标签文件地址
        '''
        self.data = []
        self.label = []
        files = os.listdir(root_path)

        # cnt = 6
        for file in tqdm(files, desc='Datasets', file=sys.stdout):
            file_path = root_path + "/" + file
            self.data.append(torch.tensor(pd.read_csv(file_path).values))  # 转化为tensor类型
            # cnt -= 1
            # if cnt == 0:
            #     break

        label_info = pd.read_csv(label_path)
        label_info = label_info[(label_info["SITE_ID"] == "NYU") & (label_info["reason"] != 2)]
        label = list(zip(label_info.group_1.values, label_info.group_2.values))
        self.label = torch.tensor(label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        '''
        Self-Attention模块，注释部分为尚未用到，代码参考Zotero上CSDN的代码
        :param num_attention_heads: 多头注意力中的头数，以老师的意思不建议多头，因为参数过多模型欠拟合
        :param input_size: Self-Attention输入层的数量，此处为时间窗的数量
        :param hidden_size:这个好像设置什么都OK，决定了QKV矩阵的尺寸，一般为N*N的就取值为num_attention_heads*input_size
        '''
        super(SelfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        # QKV三个矩阵变换
        self.query_layer = nn.Linear(input_size, self.all_head_size)
        self.key_layer = nn.Linear(input_size, self.all_head_size)
        self.value_layer = nn.Linear(input_size, self.all_head_size)

        # self.dense = nn.Linear(self.attention_head_size, self.attention_head_size)
        # self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        # self.fusion = nn.Linear(num_attention_heads, 1)

    def trans_to_multiple_head(self, x):
        new_size = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = x.to(torch.float32)
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        query_heads = self.trans_to_multiple_head(query)
        key_heads = self.trans_to_multiple_head(key)
        value_heads = self.trans_to_multiple_head(value)

        attention_scores = torch.matmul(query_heads, key_heads.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        # context = context.permute(0, 1, 3, 2)
        # context = self.fusion(context)
        # context = context.permute(0, 1, 3, 2)
        # context = context.view((1, context.size()[1], 7875))
        new_size = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_size)
        # hidden_states = self.dense(context)
        # hidden_states = self.LayerNorm(hidden_states + x)
        return context


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        # 注意力模块
        self.Attention = nn.Sequential(
            SelfAttention(Head_num, Vector_len, Vector_len * Head_num),  # self-attention的输入输出shape一样
            nn.Linear(Vector_len * Head_num, 2000),  # 6670降2000
            SelfAttention(Head_num, 2000, 2000 * Head_num),
            nn.Linear(2000 * Head_num, 200),  # 2000降200
            SelfAttention(Head_num, 200, 200 * Head_num),
            nn.Linear(200 * Head_num, 50)  # 200降50
        )

        # 展开、降维、softmax模块
        self.GetRes = nn.Sequential(
            nn.Linear(7250, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.Attention(x)
        x = x.view(x.shape[0], -1)  # 对每个样本展开成向量，7250和注意力模块最后的维度有关系，后续可能需要改一下让他自适应，暂时需要手改
        output = self.GetRes(x)
        return output


def Entire_main():
    all_data = GetData(root_path, label_path)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    module = Module()
    module = module.cuda()
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    epoch = 20  # 训练轮次

    # 画图用， 保存每一个epoch的值
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []
    p_table = PrettyTable(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "time(s)"])
    for epoch_i in range(epoch):
        # 每轮训练存储K个数据
        train_acc_list_kf = []
        train_loss_list_kf = []

        test_acc_list_kf = []
        test_loss_list_kf = []

        start_time = time.time()
        for train_index, test_index in kf.split(all_data):
            train_fold = Subset(all_data, train_index)
            test_fold = Subset(all_data, test_index)

            train_size = len(train_index)
            test_size = len(test_index)

            train_dataloader = DataLoader(train_fold, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_fold, batch_size=batch_size, shuffle=True)

            # 对于该折的所有dataloader的计量
            total_train_loss = 0
            total_train_acc = 0
            module.train()  # 设置训练模式，本身没啥用
            for data in tqdm(train_dataloader,desc=f'Epoch{epoch_i+1}',file=sys.stdout):
                x, y = data
                x = x.cuda()
                y = y.cuda()
                y = y.to(torch.float32)  # 这一步似乎很费时间
                output = module(x)

                loss = loss_fn(output, y)
                total_train_loss += loss

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
                total_train_acc += train_acc
            # 该折的训练结束，将指标传入
            train_acc_list_kf.append(float(total_train_acc / train_size))
            train_loss_list_kf.append(float(total_train_loss))

            total_test_loss = 0
            total_test_acc = 0
            module.eval()  # 设置测试模式
            with torch.no_grad():
                for data in test_dataloader:
                    x, y = data
                    x = x.cuda()
                    y = y.cuda()
                    y = y.to(torch.float32)
                    output = module(x)

                    loss = loss_fn(output, y)
                    total_test_loss += loss
                    test_acc = 0
                    for i, res in enumerate(output):
                        if res[0] > res[1]:
                            test_acc = test_acc + 1 if y[i][0] > y[i][1] else test_acc
                        if res[0] < res[1]:
                            test_acc = test_acc + 1 if y[i][0] < y[i][1] else test_acc
                    total_test_acc += test_acc
                test_acc_list_kf.append(float(total_test_acc / test_size))
                test_loss_list_kf.append(float(total_test_loss))

        # 该轮结束
        train_acc_list.append(np.mean(train_acc_list_kf))
        train_loss_list.append(np.mean(train_loss_list_kf))
        test_acc_list.append(np.mean(test_acc_list_kf))
        test_loss_list.append(np.mean(test_loss_list_kf))
        p_table.add_row(
            [epoch_i + 1, float(train_loss_list[-1]), float(train_acc_list[-1]), float(test_loss_list[-1]),
             float(test_acc_list[-1]), time.time() - start_time])
        print(p_table)

        # 传结果list格式： [train(list), test(list)]
    draw_result_pic(save_path='data166_epoch10_acc.png', res=[train_acc_list, test_acc_list], start_epoch=5,
                    pic_title='acc')
    draw_result_pic(save_path='data166_epoch10_loss.png', res=[test_acc_list, test_loss_list], start_epoch=5,
                    pic_title='loss')


if __name__ == '__main__':
    Entire_main()
