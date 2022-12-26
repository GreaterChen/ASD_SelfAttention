from torch.utils.data import Dataset, DataLoader
from utils import CheckOrder
import os
import sys
from tqdm import tqdm
import pandas as pd
import torch


class GetData(Dataset):
    def __init__(self, root_path, label_path,dataset_size):
        """
        继承于Dataset,与主函数中DataLoader一起取数据
        :param root_path: 数据集目录地址
        :param label_path: 标签文件地址
        :param dataset_size: 训练的样本总数,-1代表全部训练
        """
        self.data = []
        self.label = []
        self.files = os.listdir(root_path)  # 排一下序，确保和标签是对准的
        self.files.sort()
        self.label_info = pd.read_csv(label_path)

        self.files = self.files[:dataset_size] if dataset_size != -1 else self.files
        self.label_info = self.label_info[:dataset_size] if dataset_size != -1 else self.label_info

        CheckOrder(self.files, self.label_info)

        for file in tqdm(self.files, desc='Datasets', file=sys.stdout):
            file_path = root_path + "/" + file
            self.data.append(torch.tensor(pd.read_csv(file_path).values))  # 转化为tensor类型

        label = list(zip(self.label_info.group_1.values, self.label_info.group_2.values))

        self.label = torch.tensor(label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
