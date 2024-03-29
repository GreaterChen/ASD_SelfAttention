from requirements import *


def expand(p):
    vector = []
    row = p.shape[0]
    for i in range(row):
        vector.extend(p[i, i + 1:])
    vector = np.matrix(vector)
    return vector


def init(path):
    data_init = pd.read_csv(path)
    data = np.matrix(data_init)
    data = data[:146, :]
    row = data.shape[0]
    col = data.shape[1]
    return data, row, col


def data_dynamic(file_path, window_size, step):
    """
    对file_path文件的数据，计算一个窗口的时间点内，脑区之间的相关系数
    :param file_path: 计算的文件路径
    :param window_size: 窗口大小
    :param step: 窗口步进大小
    :return: 所有窗口的相关系数向量
    """
    data, row, col = init(file_path)
    vectors = []

    start = 0
    end = start + window_size
    while end < row:
        d = data[start:end, :]
        p = np.corrcoef(d, rowvar=False)
        vector = expand(p)
        vectors.append(vector.tolist()[0])

        start += step
        end += step

    return vectors


def data_static(file_path):
    """
    对指定文件的数据，计算所有时间点，脑区之间的相关系数向量并进行上三角展开
    :param file_path: 计算的文件路径
    :return: 相关系数向量
    """
    data, _, _ = init(file_path)
    p = np.corrcoef(data, rowvar=False)
    vector = expand(p)
    return vector


def data_static_no_expand(file_path):
    """
    对指定文件的数据，计算所有时间点，脑区之间的相关系数向量
    :param file_path: 计算的文件路径
    :return: 相关系数向量
    """
    data, _, _ = init(file_path)
    p = np.corrcoef(data, rowvar=False)
    return p


if __name__ == '__main__':
    root_path = '../raw_data/rois_aal_csv'
    files = os.listdir(root_path)
    os.mkdir('raw_data/rois_aal_csv_pearson')
    root_save_path = 'raw_data/rois_aal_csv_pearson'
    for file in tqdm(files, desc='Datasets', file=sys.stdout):
        file_path = root_path + '/' + file
        save_path = root_save_path + '/' + file
        pd.DataFrame(data_dynamic(root_path + '/' + file, 30, 1)).to_csv(save_path, index=False, header=False)
