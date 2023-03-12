import sys

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import shutil
import re


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
