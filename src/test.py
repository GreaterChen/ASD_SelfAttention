import os

import pandas as pd
import numpy as np

path = "description/label_674.csv"
label = pd.read_csv(path)
label = list(zip(label.group_1.values, label.group_2.values))
print(label)
