from requirements import *
from utils import CheckOrder

root_path = "../raw_data/rois_aal_pkl_pearson_static_expand/"
files = os.listdir(root_path)
files.sort()

label_temp = pd.read_csv("label_674.csv")

if not CheckOrder(files, label_temp):
    print("error")
    exit()
label_temp = label_temp.group_1.values

m = sum(label_temp)  # 患病人数
n = len(label_temp) - m  # 未患病人数

files_asd = []
files_hc = []
for i, item in enumerate(label_temp):
    if item == 1:
        files_asd.append(root_path + files[i])
    else:
        files_hc.append(root_path + files[i])

# 预加载所有数据，避免频繁IO
all_asd_data = []
all_hc_data = []
for file in files_asd:
    all_asd_data.append(pd.read_pickle(file))
for file in files_hc:
    all_hc_data.append(pd.read_pickle(file))
all_asd_data = pd.DataFrame(np.array(all_asd_data).reshape(len(all_asd_data), -1))
all_hc_data = pd.DataFrame(np.array(all_hc_data).reshape(len(all_hc_data), -1))
print(all_asd_data.shape, all_hc_data.shape)

tau = pd.DataFrame(columns=['ROI', 'tau'])
nc = np.zeros((6670,), dtype=int)
nd = np.zeros((6670,), dtype=int)

# 利用广播机制批量运算
for i in tqdm(range(len(all_asd_data)),desc="running",file=sys.stdout):
    ref = list(all_asd_data.iloc[i, :])
    bool_res = all_hc_data - ref > 0
    total_true = np.array(np.sum(bool_res, axis=0))
    pass
    nc += total_true
    nd += len(all_hc_data) - total_true

for i in range(6670):
    tau_t = (nc[i] - nd[i])/(m*n)
    tau = pd.DataFrame(np.insert(tau.values, len(tau.index), values=[int(i), abs(tau_t)], axis=0))

tau.columns = ['ROI', 'tau']
tau = tau.sort_values(by='tau', ascending=False)
tau = tau.reset_index(drop=True)
tau.to_csv("sort.csv",index=False)
