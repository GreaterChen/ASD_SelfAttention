import os

import pandas as pd
import numpy as np

path = "../../raw_data/rois_aal_pkl_pearson/"
files = os.listdir(path)
data = pd.read_pickle(path+files[0])
print(data.shape)
# a = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
z = np.arctanh(data)
print(z)
