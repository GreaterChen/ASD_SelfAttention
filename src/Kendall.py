import pandas as pd

from pearson_calculate import init, data_static
from requirements import *
from utils import *


def Getdata():
    root_path = '../../raw_data/rois_aal_csv'
    files = os.listdir(root_path)
    if not os.path.exists("../raw_data/rois_aal_pkl_pearson_static_expand"):
        os.makedirs("../raw_data/rois_aal_pkl_pearson_static_expand")

    for file in tqdm(files, desc='Datasets', file=sys.stdout):
        file_path = root_path + '/' + file
        save_path = "../raw_data/test"
        res = pd.DataFrame(data_static(file_path))
        res.to_pickle(save_path + '/' + file)


def calculate_kendall():
    if not os.path.exists("../Kendall_result"):
        os.makedirs("../Kendall_result")

    root_path = "../../raw_data/rois_aal_pkl_pearson/"
    files = os.listdir(root_path)
    files.sort()
    label_temp = pd.read_csv("../description/label_674.csv")

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
    for file in tqdm(files_asd, desc="读取asd数据"):
        all_asd_data.append(pd.read_pickle(file))
    for file in tqdm(files_hc, desc="读取hc数据"):
        all_hc_data.append(pd.read_pickle(file))

    # all_asd_data = pd.DataFrame(np.array(all_asd_data).reshape(len(all_asd_data), -1))
    # all_hc_data = pd.DataFrame(np.array(all_hc_data).reshape(len(all_hc_data), -1))
    # all_asd_data = pd.DataFrame(np.array(all_asd_data))

    all_asd_data_116 = []
    all_hc_data_116 = []
    for j in range(116):
        one_asd_data_116 = []
        one_hc_data_116 = []
        for i in range(len(all_asd_data)):
            one_asd_data_116.append(all_asd_data[i].iloc[j, :])
        all_asd_data_116.append(pd.DataFrame(np.array(one_asd_data_116).reshape(len(all_asd_data), -1)))
        for i in range(len(all_hc_data)):
            one_hc_data_116.append(all_hc_data[i].iloc[j, :])
        all_hc_data_116.append(pd.DataFrame(np.array(one_hc_data_116).reshape(len(all_hc_data), -1)))

    # 利用广播机制批量运算
    for j in tqdm(range(116), desc="时段", file=sys.stdout):
        all_asd_data = all_asd_data_116[j]
        all_hc_data = all_hc_data_116[j]

        tau = pd.DataFrame(columns=['ROI', 'tau'])
        nc = np.zeros((6670,), dtype=int)
        nd = np.zeros((6670,), dtype=int)

        for i in range(len(all_asd_data)):
            ref = list(all_asd_data.iloc[i, :])
            bool_res = all_hc_data - ref > 0
            total_true = np.array(np.sum(bool_res, axis=0))
            nc += total_true
            nd += len(all_hc_data) - total_true

        for i in range(6670):
            tau_t = (nc[i] - nd[i]) / (m * n)
            tau = pd.DataFrame(np.insert(tau.values, len(tau.index), values=[int(i), abs(tau_t)], axis=0))

        tau.columns = ['ROI', 'tau']
        tau = tau.sort_values(by='tau', ascending=False)
        tau = tau.reset_index(drop=True)
        tau.to_csv(f"../Kendall_result/kendall_sort_{j}.csv", index=False)


def Sort_all_116_windows():
    path = "../Kendall_result/"
    files = os.listdir(path)
    avg = np.zeros((6670,))
    for file in tqdm(files, desc="running", file=sys.stdout):
        file_path = path + file
        data = pd.read_csv(file_path)
        for i in range(6670):
            avg[data['ROI'][i]] += data['tau'][i]

    avg = avg / len(files)
    result = pd.DataFrame()
    result['ROI'] = range(6670)
    result['tau'] = avg
    result = result.sort_values(by='tau', ascending=False)
    result = result.reset_index(drop=True)
    result.to_csv("../description/Kendall_sort_all.csv")


def find_roi(end):
    """
    通过展开的特征索引反推矩阵形态坐标
    :param end: 取前多少个特征
    :return: None
    """
    data = pd.read_csv("../description/important_features_sort.csv")
    index = list(data.iloc[:end, 0])
    count = list(data.iloc[:end, 1])

    ref = np.zeros(116, )
    for i in range(116):
        ref[i] = 116 - i - 1
    for i in range(1, 116):
        ref[i] = ref[i] + ref[i - 1]

    result = []
    nums = []
    for e, feature in enumerate(index):
        for i in range(len(ref)):
            if ref[i] > feature:
                x = i
                if i == 0:
                    y = feature + i + 1
                else:
                    y = feature - ref[i - 1] + i + 1
                result.append(f"({int(x)},{int(y)})")
                nums.append(int(count[e]))
                break
    res = pd.DataFrame()
    res['坐标'] = result
    res['出现次数'] = nums
    print(res)
    res.to_csv("../description/important_roi.csv", index=False)


def explain(end):
    """
    获取出现在前100的特征以及出现的次数
    :param end: 表示出现在每个窗口的前end个特征
    :return: None
    """
    count = np.zeros(6670, )
    path = "../../Kendall_result/"
    files = os.listdir(path)

    for file in files:
        data = pd.read_csv(path + file)
        for i in list(data.iloc[:end, :]['ROI']):
            count[i] += 1

    roi = []
    num = []
    result = pd.DataFrame()
    for index, i in enumerate(count):
        if i != 0:
            roi.append(index)
            num.append(i)
    result['ROI'] = roi
    result['count'] = num
    result = result.sort_values(by='count', ascending=False)
    result.to_csv("../description/important_features_sort.csv", index=False)
    find_roi(end)


if __name__ == '__main__':
    explain(30)
