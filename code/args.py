# 超参数
dataset_size = -1  # 训练的样本总数,-1代表全部训练,调试的时候可以改小点
batch_size = 2  # 每次训练样本数
Head_num = 1  # self-attention的头数
epoch = 200  # 最多训练轮次，极大概率会早停，这个设大点也ok
learn_rate = 0.001  # 学习率

# 其他设置
# root_path = "../raw_data/rois_aal_pkl_pearson"
root_path = "/root/autodl-tmp/rois_aal_pkl_pearson"
label_path = "label_674.csv"

num_workers = 15    # datalodaer的线程数
pre_train = False    # 是否采用预训练模型

EarlyStop = True    # 是否采用早停策略
EarlyStop_patience = 10  # 能容忍多少次测试集损失值无下降
EarlyStop_epoch = 50    # 从多少轮开始启用早停策略（若刚开始就使用可能会导致过早的训练停止）

Flood = False    # 是否采用Flood(理解原理后再尝试使用)
flood_value = 140   # Flood中的超参数

# 常量
Windows_num = 116  # 时间窗的个数
Vector_len = int(116 * 115 / 2)  # 上三角展开后的长度
data_num = -1  # 数据集个数(自动获取)





