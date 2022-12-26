from torch.utils.data import Dataset, DataLoader
import os
import sys
from tqdm import tqdm
import pandas as pd
import torch


class GetData(Dataset):
    def __init__(self, root_path, label_path):
        '''
        继承于Dataset,与主函数中DataLoader一起取数据
        :param root_path: 数据集目录地址
        :param label_path: 标签文件地址
        '''
        self.data = []
        self.label = []
        self.files = os.listdir(root_path)

        for file in tqdm(self.files, desc='Datasets', file=sys.stdout):
            file_path = root_path + "/" + file
            self.data.append(torch.tensor(pd.read_csv(file_path).values))  # 转化为tensor类型

        label_info = pd.read_csv(label_path)
        label_info = label_info[(label_info["SITE_ID"] == "NYU") & (label_info["reason"] != 2)]
        label = list(zip(label_info.group_1.values, label_info.group_2.values))
        self.label = torch.tensor(label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
