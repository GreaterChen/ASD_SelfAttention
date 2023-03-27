import shutil

from requirements import *

raw_path = r"D:\study\ASD_others\raw_data\cc200_noextend"
ref_path = r"D:\study\ASD_others\raw_data\cc200_extend_train"
throw_path = r"D:\study\ASD_others\raw_data\cc200_noextend_throw"
files = os.listdir(raw_path)
ref_files = os.listdir(ref_path)
for file in files:
    if file not in ref_files:
        shutil.move(raw_path + '/' + file, throw_path + '/' + file)
