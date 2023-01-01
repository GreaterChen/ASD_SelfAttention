import os
import sys

import pandas as pd
from tqdm import tqdm

root_path = "../raw_data/rois_aal_pkl_pearson/"
files = os.listdir(root_path)

for file in tqdm(files, desc="修复进度", file=sys.stdout):
    data = pd.read_pickle(root_path + file)
    data = data.T.reset_index().rename(columns={'index': '-1'}).T
    data = pd.DataFrame(data, dtype="float64")
    data.to_pickle(root_path + file)
