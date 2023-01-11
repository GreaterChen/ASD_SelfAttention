import pandas as pd

from pearson_calculate import init, data_static
from requirements import *

root_path = '../../rois_aal_csv'
files = os.listdir(root_path)
if not os.path.exists("../raw_data/rois_aal_pkl_pearson_static_expand"):
    os.makedirs("../raw_data/rois_aal_pkl_pearson_static_expand")
for file in tqdm(files, desc='Datasets', file=sys.stdout):
    file_path = root_path + '/' + file
    save_path = "../raw_data/rois_aal_pkl_pearson_static_expand"
    res = pd.DataFrame(data_static(file_path))
    # res.to_pk(save_path + '/' + file, index=False, header=False)
    res.to_pickle(save_path + '/' + file)
