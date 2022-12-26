import os
import pandas as pd


def check_order(files, label):
    label_name = label['SUB_ID'].to_list()
    error_sign = 1
    for i in range(len(label_name)):
        if str(label_name[i]) not in str(files[i]):
            error_sign = 0

    if error_sign:
        print("数据、标签已对准")
    else:
        print("数据、标签未对准！请检查！")
