# 超参数
dataset_size = -1  # 训练的样本总数,-1代表全部训练,调试的时候可以改小点
batch_size = 2  # 每次训练样本数
Head_num = 1  # self-attention的头数
epoch = 100  # 最多训练轮次
learn_rate = 0.001  # 学习率

num_works = 12  # datalodaer的线程数

# 常量
Windows_num = 115  # 时间窗的个数
Vector_len = int(116 * 115 / 2)  # 上三角展开后的长度
data_num = -1  # 数据集个数(自动获取)


root_path = "/root/autodl-tmp/rois_aal_pkl_pearson"
label_path = "label_674.csv"
