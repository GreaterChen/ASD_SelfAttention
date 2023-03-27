import sys

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import shutil
import re
import scipy.io as io


def select_cc200():
    linux_path = "/root/autodl-tmp/cc200_extend/"

    refer_path = "/root/autodl-tmp/rois_aal_pkl_pearson/"
    print(len(refer_path))
    files = os.listdir(linux_path)
    refer_files = os.listdir(refer_path)
    # print(refer_files)
    refer_nums = []
    for item in refer_files:
        refer_nums.append(re.findall(r'\d{7}', item)[0])
    exist_files = []
    for file in tqdm(files, desc="running", file=sys.stdout):
        nums = re.findall(r'\d{7}', file)[0]
        if nums in refer_nums:
            exist_files.append(file)
    print(exist_files)
    print(len(exist_files))

    os.makedirs("/root/autodl-tmp/cc200_extend_train/")
    for file in exist_files:
        before_path = linux_path + file
        after_path = "/root/autodl-tmp/cc200_extend_train/"
        shutil.move(before_path, after_path + file)


def del_useless():
    path = "/root/autodl-tmp/cc200_extend_train/"
    files = os.listdir(path)
    shape = []
    for file in files:
        data = pd.read_pickle(path + file)
        if data.shape[0] not in shape:
            shape.append(data.shape[0])

    print(shape)
    # shape = []
    # exist = []
    # remove = []
    # shorten = []
    # cnt = 0
    # if not os.path.exists(r"D:\study\ASD_others\raw_data\cc200_extend_train_throw/"):
    #     os.makedirs(r"D:\study\ASD_others\raw_data\cc200_extend_train_throw/")
    # for file in tqdm(files, desc="running", file=sys.stdout):
    #     data = pd.read_pickle(path + file)
    #     # if data.shape[0] < 116:
    #     #     remove.append(file)
    #     if data.shape[0] > 116:
    #         data = data.iloc[:116, :]
    #         data.to_pickle(path + file)
    #         shorten.append(file)
    # for file in remove:/
    #     before_path = path + file
    #     after_path = r"D:\study\ASD_others\raw_data\cc200_extend_train_throw/"
    #     shutil.move(before_path, after_path + file)

    # data = pd.read_pickle(r"D:\study\ASD_others\raw_data\cc200_extend_train_throw\UM_1_0050272_rois_cc200.pkl")
    # print(data)


if __name__ == '__main__':
    data = io.loadmat('description/CenterOfMass_AAL.mat')

    print(data.keys())
